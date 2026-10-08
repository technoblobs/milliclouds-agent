#!/usr/bin/env python3
"""Renderyard GPU agent.

Modes (env-selected):
  One-shot:  RY_API, RY_JOB, RY_TOKEN            — render claimed frame range, exit
  Worker:    RY_API, RY_WORKER_TOKEN, WORKER_MODE=1 — poll chunks forever (unless drained)

Outbound HTTPS only. Frame-level chunks are claimed via the API; Redis/logs stay private.
"""
from __future__ import annotations

import os
import platform
import json
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

API = os.environ.get("RY_API", "").rstrip("/")
POLL_INTERVAL = int(os.environ.get("RY_POLL_INTERVAL", "10"))
HEARTBEAT_INTERVAL = 20
WORKER_HB_INTERVAL = 25
LOG_FLUSH_INTERVAL = float(os.environ.get("RY_LOG_FLUSH_INTERVAL", "1.0"))
LOG_BATCH_MAX = 80
MAX_EXTRACT_BYTES = 20 * 1024**3
MAX_EXTRACT_FILES = 20_000
MAX_LOG_LINE = 2000
BLENDER = os.environ.get("BLENDER_BIN", "blender")
AGENT_VERSION = "1.2.0"
RENDER_MODE = os.environ.get("RY_RENDER_MODE", "secure").lower()
SANDBOX_ROOT = Path(os.environ.get("RY_SANDBOX_ROOT", "/sandbox"))
RENDERER_MODE = os.environ.get("RY_RENDERER_MODE", "").lower() in ("1", "true", "yes")
RENDER_TIMEOUT_SECONDS = int(os.environ.get("RY_RENDER_TIMEOUT_SECONDS", str(12 * 60 * 60)))
CONTROL_PORT = int(os.environ.get("RY_CONTROL_PORT", "0"))
CONTROL_TOKEN = os.environ.get("RY_CONTROL_TOKEN", "")

_stop = threading.Event()
_abort_render = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: _stop.set())
signal.signal(signal.SIGINT, lambda *_: _stop.set())


def abort_current_render(reason: str) -> None:
    """Kill the in-flight Blender job; worker process stays up."""
    if not _abort_render.is_set():
        log(f"abort render: {reason}")
    _abort_render.set()


def render_should_stop() -> bool:
    return _stop.is_set() or _abort_render.is_set()

_log_shipper: "LogShipper | None" = None
_worker_log_shipper: "WorkerLogShipper | None" = None
_run_policy = "accept_jobs"  # updated from API
_runtime_metrics: dict = {}
_status: dict = {
    "mode": RENDER_MODE,
    "presence": "starting",
    "job_id": None,
    "chunk_id": None,
    "frames_done": 0,
    "last_error": None,
}


def log(msg: str) -> None:
    line = f"[agent] {msg}"
    print(line, flush=True)
    if _log_shipper is not None:
        _log_shipper.add(line)
    elif _worker_log_shipper is not None:
        _worker_log_shipper.add(line)


def _control_status() -> dict:
    return {
        **_status,
        "agent_version": AGENT_VERSION,
        "render_mode": RENDER_MODE,
        "metrics": dict(_runtime_metrics),
        "stop_requested": _stop.is_set(),
        "abort_render": _abort_render.is_set(),
    }


class ControlHandler(BaseHTTPRequestHandler):
    """Optional localhost dashboard; disabled unless RY_CONTROL_PORT is set."""

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return not CONTROL_TOKEN or self.headers.get("X-RY-Control-Token") == CONTROL_TOKEN

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/" and CONTROL_TOKEN and not self._authorized():
            self._send(403, "text/plain", b"forbidden")
            return
        if self.path == "/api/status":
            self._send(200, "application/json", json.dumps(_control_status()).encode())
            return
        if self.path == "/":
            body = b"""<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>
<title>MilliCloud Agent</title><style>body{font:14px monospace;background:#0c0f14;color:#e8ecf2;max-width:800px;margin:40px auto;padding:0 20px}pre{white-space:pre-wrap;background:#151a22;padding:16px;border-radius:8px}button{background:#ffb454;border:0;padding:10px 14px;border-radius:5px;cursor:pointer;margin-right:8px}</style>
<h1>MilliCloud Agent</h1><p>Local diagnostics. Abort discards the current chunk; finished chunks stay.</p>
<button onclick='abortRender()'>Abort current render</button>
<button onclick='stopAgent()'>Stop agent</button><pre id=s>Loading...</pre>
<script>let token=decodeURIComponent(location.hash.slice(1));if(!token)token=prompt('Control token (leave blank if not configured)')||'';const headers=()=>token?{'X-RY-Control-Token':token}:{};async function load(){const r=await fetch('/api/status',{headers:headers()});document.querySelector('#s').textContent=JSON.stringify(await r.json(),null,2)}async function stopAgent(){await fetch('/api/stop',{method:'POST',headers:headers()});load()}async function abortRender(){if(!confirm('Abort this render? In-progress frames on this machine will be discarded.'))return;await fetch('/api/abort-render',{method:'POST',headers:headers()});load()}load();setInterval(load,2000)</script>"""
            self._send(200, "text/html; charset=utf-8", body)
            return
        self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send(403, "text/plain", b"forbidden")
            return
        if self.path == "/api/stop":
            _stop.set()
            self._send(202, "application/json", b'{"detail":"stop requested"}')
            return
        if self.path == "/api/abort-render":
            abort_current_render("local control panel")
            self._send(202, "application/json", b'{"detail":"abort requested"}')
            return
        self._send(404, "text/plain", b"not found")

    def log_message(self, *_args) -> None:
        return


def start_control_panel() -> None:
    """Optional loopback diagnostics. Never required to poll/render. Must not kill the worker."""
    if CONTROL_PORT <= 0:
        return
    last_err: OSError | None = None
    for port in (CONTROL_PORT, *range(CONTROL_PORT + 1, CONTROL_PORT + 8)):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), ControlHandler)
            threading.Thread(target=server.serve_forever, name="control-panel", daemon=True).start()
            if port != CONTROL_PORT:
                log(f"control port {CONTROL_PORT} busy — panel on 127.0.0.1:{port}")
            else:
                log(f"local control panel on 127.0.0.1:{port}")
            return
        except OSError as e:
            last_err = e
            continue
    log(
        f"control panel disabled (ports {CONTROL_PORT}–{CONTROL_PORT + 7} in use: {last_err}). "
        "The worker does not need this port; API stays on RY_API."
    )


def api(method: str, path: str, token: str, **kw) -> requests.Response:
    return requests.request(
        method, f"{API}{path}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
        **kw,
    )


# ---------- error classification (reported to API / UI) ----------

def classify_error(exc: BaseException) -> str:
    s = str(exc).lower()
    if any(x in s for x in ("no space", "enospc", "disk full", "disk quota", "no space left")):
        return "disk_full"
    if any(x in s for x in ("cuda", "optix", "hip ", "gpu", "nvidia", "out of memory", "oom")):
        return "gpu_error"
    if "blender" in s and ("exited" in s or "crash" in s or "code" in s):
        return "blender_crash"
    if "openfoam" in s or "simplefoam" in s or "icofoam" in s or "blockmesh" in s or "snappy" in s:
        return "openfoam_failed"
    if s.startswith("no_gpu") or "no cuda or apple metal" in s:
        return "no_gpu"
    if s.startswith("no_vram") or "free vram" in s:
        return "no_vram"
    if s.startswith("no_disk") or s.startswith("no_libs") or "pytorch is not installed" in s:
        return "no_disk" if "disk" in s or "no_disk" in s else "no_libs"
    if "flux" in s or "image job" in s or s.startswith("image_failed") or s.startswith("model_load"):
        return "image_failed"
    if "lora" in s or s.startswith("lora_failed"):
        return "lora_failed"
    if "unsafe path" in s or "extraction" in s or "archive" in s:
        return "scene_error"
    if "lease" in s or "reassigned" in s or "409" in s:
        return "lease_lost"
    if "abort" in s or "cancelled" in s or "operator" in s:
        return "cancelled"
    return "render_failed"


# ---------- host specs (best-effort; never fails the agent) ----------

def detect_gpu() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except Exception:
        return None


def collect_specs() -> dict:
    specs: dict = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "arch": platform.machine(),
        "hostname": socket.gethostname(),
        "agent_version": AGENT_VERSION,
    }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 4:
                specs["gpu_name"] = parts[0]
                specs["vram_mb"] = int(float(parts[1])) if parts[1] else None
                specs["driver"] = parts[2]
                specs["gpu_util_pct"] = int(float(parts[3])) if parts[3] else None
    except Exception:
        pass
    try:
        usage = shutil.disk_usage("/")
        specs["disk_total_gb"] = round(usage.total / 1024**3, 1)
        specs["disk_free_gb"] = round(usage.free / 1024**3, 1)
    except Exception:
        pass
    try:
        # Linux meminfo
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    specs["ram_mb"] = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass
    try:
        out = subprocess.run([BLENDER, "--version"], capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout:
            specs["blender"] = out.stdout.strip().splitlines()[0][:80]
    except Exception:
        pass
    return specs


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def collect_runtime_metrics() -> dict:
    """Read local cgroup/GPU metrics without requiring Docker socket access."""
    metrics: dict = {}
    cgroup = Path("/sys/fs/cgroup")
    memory_current = _read_int(cgroup / "memory.current")
    memory_max = _read_int(cgroup / "memory.max")
    pids_current = _read_int(cgroup / "pids.current")
    if memory_current is not None:
        metrics["memory_used_mb"] = round(memory_current / 1024**2, 1)
    if memory_max is not None and memory_max < 2**63:
        metrics["memory_limit_mb"] = round(memory_max / 1024**2, 1)
    if pids_current is not None:
        metrics["processes"] = pids_current
    try:
        cpu_stat = (cgroup / "cpu.stat").read_text().splitlines()
        cpu = dict(line.split(maxsplit=1) for line in cpu_stat if " " in line)
        if "usage_usec" in cpu:
            metrics["cpu_usage_seconds"] = round(int(cpu["usage_usec"]) / 1_000_000, 2)
        if "nr_throttled" in cpu:
            metrics["cpu_throttled_events"] = int(cpu["nr_throttled"])
    except (OSError, ValueError):
        pass
    try:
        usage = shutil.disk_usage(SANDBOX_ROOT if SANDBOX_ROOT.exists() else "/")
        metrics["disk_free_gb"] = round(usage.free / 1024**3, 1)
        metrics["disk_used_pct"] = round((usage.total - usage.free) / max(1, usage.total) * 100, 1)
    except OSError:
        pass
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            values = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
            if len(values) >= 5:
                metrics["gpu_util_pct"] = int(float(values[0]))
                metrics["gpu_memory_used_mb"] = int(float(values[1]))
                metrics["gpu_memory_total_mb"] = int(float(values[2]))
                metrics["gpu_temperature_c"] = int(float(values[3]))
                metrics["gpu_power_w"] = round(float(values[4]), 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return metrics


def render_env(workdir: Path) -> dict[str, str]:
    """Environment for Blender. Never pass agent/API tokens.

    Trusted/desktop: inherit the host env (Metal/CUDA need it). A stripped env
    on macOS often SIGABRTs Blender with no useful log.
    Secure: keep the tight allowlist.
    """
    tmp = workdir / "tmp"
    scripts = workdir / "empty-scripts"
    tmp.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)

    drop = {k for k in os.environ if k.startswith("RY_") or k in ("RY_WORKER_TOKEN", "WORKER_MODE")}
    if RENDER_MODE == "trusted":
        env = {k: v for k, v in os.environ.items() if k not in drop}
        env["TMPDIR"] = str(tmp)
        env["BLENDER_USER_SCRIPTS"] = str(scripts)
        return env

    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workdir / "home"),
        "TMPDIR": str(tmp),
        "BLENDER_USER_CONFIG": str(workdir / "blender-config"),
        "BLENDER_USER_SCRIPTS": str(scripts),
    }
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["BLENDER_USER_CONFIG"]).mkdir(parents=True, exist_ok=True)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XAUTHORITY",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
    ):
        if os.environ.get(name):
            env[name] = os.environ[name]
    return env


def emit_blender(line: str, event_sink=None) -> None:
    """Blender stdout → desktop agent log + job log shipper."""
    if not line:
        return
    print(f"[blender] {line}", flush=True)
    if _log_shipper is not None:
        _log_shipper.add(f"[blender] {line}")
    if event_sink is not None:
        event_sink({"type": "log", "text": line})


# ---------- live log shipping ----------

class LogShipper:
    """Job/chunk Blender + agent lines → POST /agent/jobs/{id}/logs."""

    def __init__(self, job_id: str, token: str, chunk_id: str | None = None):
        self.job_id = job_id
        self.token = token
        self.chunk_id = chunk_id
        self._lock = threading.Lock()
        self._buf: list[tuple[int, str]] = []
        self._next_seq = 1
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="log-shipper", daemon=True)
        self._lost_lease = False

    def start(self) -> None:
        self._thread.start()

    def add(self, text: str) -> None:
        if self._done.is_set() or self._lost_lease:
            return
        line = text.rstrip("\n\r")
        if not line:
            return
        if len(line) > MAX_LOG_LINE:
            line = line[: MAX_LOG_LINE - 1] + "…"
        with self._lock:
            self._buf.append((self._next_seq, line))
            self._next_seq += 1
            if len(self._buf) > 5_000:
                self._buf = self._buf[-2_500:]

    def _loop(self) -> None:
        while not self._done.wait(LOG_FLUSH_INTERVAL):
            self.flush()
        self.flush()

    def flush(self) -> None:
        if self._lost_lease:
            return
        with self._lock:
            if not self._buf:
                return
            batch = self._buf[:LOG_BATCH_MAX]
            self._buf = self._buf[LOG_BATCH_MAX:]
        seq_start = batch[0][0]
        lines = [t for _, t in batch]
        payload: dict = {"lines": lines, "seq_start": seq_start}
        if self.chunk_id:
            payload["chunk_id"] = self.chunk_id
        try:
            r = api("POST", f"/agent/jobs/{self.job_id}/logs", self.token, json=payload)
            if r.status_code == 409:
                print("[agent] log shipper: job no longer active", flush=True)
                self._lost_lease = True
                with self._lock:
                    self._buf.clear()
                return
            if r.status_code == 429:
                with self._lock:
                    self._buf = batch + self._buf
                time.sleep(2)
                return
            if r.status_code >= 400:
                if r.status_code in (401, 403):
                    self._lost_lease = True
                    return
                with self._lock:
                    self._buf = batch + self._buf
        except requests.RequestException:
            with self._lock:
                self._buf = batch + self._buf

    def close(self) -> None:
        self._done.set()
        self._thread.join(timeout=10)
        self.flush()


class WorkerLogShipper:
    """Host lifecycle lines → POST /agent/worker/logs (idle polls, drain, claim events)."""

    def __init__(self, token: str):
        self.token = token
        self._lock = threading.Lock()
        self._buf: list[tuple[int, str]] = []
        self._next_seq = 1
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="worker-log", daemon=True)
        self.job_id: str | None = None
        self.chunk_id: str | None = None

    def start(self) -> None:
        self._thread.start()

    def set_context(self, job_id: str | None = None, chunk_id: str | None = None) -> None:
        self.job_id = job_id
        self.chunk_id = chunk_id

    def add(self, text: str) -> None:
        if self._done.is_set():
            return
        line = text.rstrip("\n\r")
        if not line:
            return
        if len(line) > MAX_LOG_LINE:
            line = line[: MAX_LOG_LINE - 1] + "…"
        with self._lock:
            self._buf.append((self._next_seq, line))
            self._next_seq += 1
            if len(self._buf) > 2_000:
                self._buf = self._buf[-1_000:]

    def _loop(self) -> None:
        while not self._done.wait(LOG_FLUSH_INTERVAL * 2):
            self.flush()
        self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buf:
                return
            batch = self._buf[:LOG_BATCH_MAX]
            self._buf = self._buf[LOG_BATCH_MAX:]
        seq_start = batch[0][0]
        lines = [t for _, t in batch]
        payload: dict = {"lines": lines, "seq_start": seq_start}
        if self.job_id:
            payload["job_id"] = self.job_id
        if self.chunk_id:
            payload["chunk_id"] = self.chunk_id
        try:
            r = api("POST", "/agent/worker/logs", self.token, json=payload)
            if r.status_code in (401, 403):
                with self._lock:
                    self._buf.clear()
                return
            if r.status_code >= 400:
                with self._lock:
                    self._buf = batch + self._buf
        except requests.RequestException:
            with self._lock:
                self._buf = batch + self._buf

    def close(self) -> None:
        self._done.set()
        self._thread.join(timeout=5)
        self.flush()


# ---------- scene / render ----------

def download_scene(url: str, dest: Path) -> None:
    log(f"downloading scene -> {dest.name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)


def safe_extract(zip_path: Path, dest: Path) -> None:
    total, count = 0, 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            count += 1
            total += info.file_size
            if count > MAX_EXTRACT_FILES or total > MAX_EXTRACT_BYTES:
                raise RuntimeError("Archive exceeds extraction limits")
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {info.filename}")
        z.extractall(dest)


def find_blend(root: Path) -> Path:
    blends = sorted(root.rglob("*.blend"))
    if not blends:
        raise RuntimeError("No .blend file found in the archive")
    return blends[0]


def find_foam_case(root: Path) -> Path:
    hits = sorted(root.rglob("system/controlDict"))
    if not hits:
        raise RuntimeError(
            "No OpenFOAM case in the archive (expected system/controlDict next to 0/ and constant/)"
        )
    return hits[0].parent.parent


def parse_control_dict(path: Path) -> dict:
    text = path.read_text(errors="replace")
    def grab(key: str, default: str | None = None) -> str | None:
        m = re.search(rf"(?m)^\s*{re.escape(key)}\s+([^\s;]+)\s*;", text)
        return m.group(1) if m else default
    def num(key: str, default: float) -> float:
        raw = grab(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default
    return {
        "application": grab("application") or "simpleFoam",
        "startTime": num("startTime", 0.0),
        "endTime": num("endTime", 1.0),
    }


def foam_source_prefix() -> str:
    return (
        "source /usr/lib/openfoam/openfoam2412/etc/bashrc 2>/dev/null || "
        "source /usr/lib/openfoam/openfoam2406/etc/bashrc 2>/dev/null || "
        "source /opt/openfoam11/etc/bashrc 2>/dev/null || "
        "source /opt/openfoam12/etc/bashrc 2>/dev/null || true; "
    )


def _foam_args_for_container(args: list[str]) -> list[str]:
    """Drop host -case paths; the container cwd is the case (/work)."""
    out: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "-case":
            skip = True
            continue
        out.append(a)
    return out


def foam_invoke(args: list[str], case: Path, image: str) -> list[str]:
    """Native OpenFOAM if on PATH, otherwise Docker.

    Mount the case at /work. Do not pass macOS /var/folders paths into the
    container — OpenFOAM then fails with 'cannot open root directory'.
    """
    if shutil.which(args[0]):
        stdbuf = shutil.which("stdbuf")
        return ([stdbuf, "-oL", "-eL", *args] if stdbuf else args)
    if not shutil.which("docker"):
        raise RuntimeError(
            "OpenFOAM is not installed and docker was not found. "
            "Install OpenFOAM or Docker (opencfd/openfoam-default)."
        )
    inner = _foam_args_for_container(args)
    quoted = " ".join(shlex.quote(a) for a in inner)
    cmd = ["docker", "run", "--rm"]
    if platform.machine().lower() in ("arm64", "aarch64"):
        cmd += ["--platform", "linux/amd64"]
    cmd += [
        "-v", f"{case.resolve()}:/work",
        "-w", "/work",
        image,
        "bash", "-lc",
        # stdbuf: OpenFOAM fully-buffers stdout when there is no TTY.
        foam_source_prefix() + "cd /work && stdbuf -oL -eL " + quoted,
    ]
    return cmd


def patch_end_time(control: Path, end_time: float) -> None:
    text = control.read_text(errors="replace")
    new, n = re.subn(r"(?m)^(\s*endTime\s+)[^\s;]+(\s*;)", rf"\g<1>{end_time}\2", text, count=1)
    if n:
        control.write_text(new)


def foam_time_dirs(case: Path) -> list[Path]:
    out: list[tuple[float, Path]] = []
    for p in case.iterdir():
        if not p.is_dir() or p.name in ("0.org",):
            continue
        try:
            t = float(p.name)
        except ValueError:
            continue
        out.append((t, p))
    out.sort()
    return [p for _, p in out]


def pack_foam_results(case: Path, dest_zip: Path, log_path: Path | None) -> Path:
    times = foam_time_dirs(case)
    last = times[-1] if times else None
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        def add_tree(root: Path, arc_prefix: str) -> None:
            if not root.exists():
                return
            if root.is_file():
                zf.write(root, arc_prefix)
                return
            for f in root.rglob("*"):
                if f.is_file():
                    zf.write(f, str(Path(arc_prefix) / f.relative_to(root)))

        if last:
            add_tree(last, last.name)
        post = case / "postProcessing"
        if post.exists():
            add_tree(post, "postProcessing")
        if log_path and log_path.exists():
            zf.write(log_path, log_path.name)
        ctrl = case / "system" / "controlDict"
        if ctrl.exists():
            zf.write(ctrl, "system/controlDict")
    if dest_zip.stat().st_size < 32:
        raise RuntimeError("OpenFOAM produced no packable results")
    return dest_zip


_TIME_RE = re.compile(r"^Time\s*=\s*([0-9.eE+-]+)")


def run_openfoam(case: Path, job: dict, on_progress) -> list[Path]:
    settings = job.get("engine_settings") or {}
    image = settings.get("docker_image") or os.environ.get(
        "OPENFOAM_DOCKER_IMAGE", "opencfd/openfoam-default:2412"
    )
    ctrl_path = case / "system" / "controlDict"
    if not ctrl_path.exists():
        raise RuntimeError("OpenFOAM case is missing system/controlDict")
    meta = parse_control_dict(ctrl_path)
    solver = settings.get("solver") or "auto"
    if solver in (None, "", "auto"):
        solver = meta["application"]
    if settings.get("end_time") is not None:
        patch_end_time(ctrl_path, float(settings["end_time"]))
        meta = parse_control_dict(ctrl_path)
    start_t, end_t = meta["startTime"], meta["endTime"]
    nprocs = int(settings.get("nprocs") or 1)
    log(f"openfoam solver={solver} endTime={end_t} nprocs={nprocs}")

    def restore_zero() -> None:
        """motorBike etc. keep fields in 0.orig; the solver needs 0/p, 0/U, …"""
        orig = case / "0.orig"
        zero = case / "0"
        if orig.is_dir() and not (zero / "p").exists():
            if zero.exists():
                shutil.rmtree(zero)
            shutil.copytree(orig, zero)
            log("copied 0.orig → 0 (initial fields)")

    restore_zero()
    if not (case / "0" / "p").exists() and not (case / "0.orig" / "p").exists():
        raise RuntimeError(
            "OpenFOAM case has no 0/p or 0.orig/p. Re-zip including the 0/ and 0.orig/ folders."
        )

    steps: list[list[str]] = []
    if settings.get("run_block_mesh", True) and (case / "system" / "blockMeshDict").exists():
        steps.append(["blockMesh", "-case", str(case)])
    if (case / "system" / "surfaceFeatureExtractDict").exists():
        steps.append(["surfaceFeatureExtract", "-case", str(case)])
    if settings.get("run_snappy", True) and (case / "system" / "snappyHexMeshDict").exists():
        geom = case / "constant" / "triSurface"
        geom2 = case / "constant" / "geometry"
        if not any(p.exists() and any(p.iterdir()) for p in (geom, geom2) if p.exists()):
            log("warning: snappyHexMeshDict present but no constant/triSurface geometry")
        steps.append(["snappyHexMesh", "-overwrite", "-case", str(case)])
    if (case / "system" / "topoSetDict").exists():
        steps.append(["topoSet", "-case", str(case)])
    if (case / "system" / "createPatchDict").exists():
        steps.append(["createPatch", "-overwrite", "-case", str(case)])
    if nprocs > 1:
        steps.append(["decomposePar", "-case", str(case)])
        steps.append(["mpirun", "-np", str(nprocs), solver, "-parallel", "-case", str(case)])
        steps.append(["reconstructPar", "-latestTime", "-case", str(case)])
    else:
        steps.append([solver, "-case", str(case)])

    log_path = case / f"log.{solver}"
    log_fh = open(log_path, "w", encoding="utf-8")
    try:
        for step in steps:
            if render_should_stop():
                raise RuntimeError("Render aborted by operator")
            log(f"openfoam exec: {' '.join(step[:4])}…")
            is_solver = step[0] == solver or (len(step) > 3 and solver in step)
            if is_solver:
                restore_zero()
                if not (case / "0" / "p").exists():
                    raise RuntimeError(
                        "No 0/p after meshing. Include 0.orig/ in the zip (motorBike tutorial)."
                    )
            cmd = foam_invoke(step, case, image)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        print(f"[openfoam] {line}", flush=True)
                        if _log_shipper is not None:
                            _log_shipper.add(f"[openfoam] {line}")
                        log_fh.write(line + "\n")
                        log_fh.flush()
                    if is_solver:
                        m = _TIME_RE.match(line)
                        if m:
                            t = float(m.group(1))
                            span = max(1e-12, end_t - start_t)
                            on_progress(max(0.0, min(0.99, (t - start_t) / span)))
                    if render_should_stop():
                        proc.terminate()
                        raise RuntimeError("Render aborted by operator")
            finally:
                proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"{step[0]} exited with code {proc.returncode}")
    finally:
        log_fh.close()

    dest = case.parent / "openfoam-results.zip"
    pack_foam_results(case, dest, log_path)
    log(f"packed results {dest.name} ({dest.stat().st_size} bytes)")
    return [dest]


IMAGE_MODEL_IDS = {
    "auto": "auto",
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
}
DEFAULT_IMAGE_DOCKER = os.environ.get("IMAGE_DOCKER_IMAGE", "millicloud-image:klein")


def image_generate_script() -> Path:
    env = os.environ.get("IMAGE_GENERATE")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for p in (
        here.parent / "image" / "generate.py",
        here / "generate.py",
        Path("/opt/image/generate.py"),
    ):
        if p.is_file():
            return p
    return here.parent / "image" / "generate.py"


def image_python() -> str | None:
    env = os.environ.get("IMAGE_PYTHON")
    if env and Path(env).is_file():
        return env
    here = Path(__file__).resolve().parent
    venv = here.parent / "image" / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return None


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def run_image(claim: dict, workdir: Path, init_image: Path | None, on_progress) -> list[Path]:
    """txt2img / img2img via generate.py (native MPS or NVIDIA Docker)."""
    settings = claim.get("engine_settings") or {}
    mode = str(settings.get("mode") or "txt2img").lower()
    if mode == "img2img" and (init_image is None or not init_image.is_file()):
        raise RuntimeError("img2img requires a reference image")
    out_dir = workdir / "out"
    out_dir.mkdir(exist_ok=True)
    model_key = str(settings.get("model") or "auto")
    model_id = settings.get("model_id") or IMAGE_MODEL_IDS.get(model_key) or "auto"
    script = image_generate_script()
    py = image_python()
    use_docker = bool(has_nvidia() and shutil.which("docker"))
    # Prefer native venv on Mac even if docker exists (Metal).
    if sys.platform == "darwin" and py:
        use_docker = False

    # Docker mounts workdir at /work — job JSON must use container paths.
    if use_docker:
        out_for_job = "/work/out"
        init_for_job = None
        if init_image is not None:
            dest = workdir / f"init{init_image.suffix.lower() or '.png'}"
            if init_image.resolve() != dest.resolve():
                shutil.copy2(init_image, dest)
            init_for_job = f"/work/{dest.name}"
    else:
        out_for_job = str(out_dir)
        init_for_job = str(init_image) if init_image is not None else None

    job = {
        "mode": mode,
        "prompt": settings.get("prompt") or "",
        "negative_prompt": settings.get("negative_prompt") or "",
        "width": int(settings.get("width") or 1024),
        "height": int(settings.get("height") or 1024),
        "steps": int(settings.get("steps") or 4),
        "guidance": float(settings.get("guidance") or 1.0),
        "strength": float(settings.get("strength") or 0.75),
        "num_images": int(settings.get("num_images") or 1),
        "seed": int(settings.get("seed") if settings.get("seed") is not None else -1),
        "model_id": model_id,
        "output_dir": out_for_job,
    }
    if init_for_job:
        job["init_image"] = init_for_job
    job_path = workdir / "image-job.json"
    job_path.write_text(json.dumps(job, indent=2))
    if use_docker:
        image = settings.get("docker_image") or DEFAULT_IMAGE_DOCKER
        cache = Path(os.environ.get("IMAGE_CACHE") or (Path.home() / ".cache" / "millicloud-image"))
        cache.mkdir(parents=True, exist_ok=True)
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{workdir.resolve()}:/work",
            "-v", f"{cache.resolve()}:/cache",
            "-e", "HF_HOME=/cache/hf",
            "-e", "PYTHONUNBUFFERED=1",
        ]
        if script.is_file():
            cmd += ["-v", f"{script.resolve()}:/opt/image/generate.py:ro"]
        cmd += [
            image,
            "--job", "/work/image-job.json",
        ]
        log(f"image docker {image} mode={mode}")
    else:
        if not py:
            raise RuntimeError(
                "no_libs: Image runner not set up. On a Mac run image/setup-macos.sh. "
                "On NVIDIA Linux build image/Dockerfile as millicloud-image:klein "
                "(see image.md)."
            )
        if not script.is_file():
            raise RuntimeError(f"no_libs: generate.py not found at {script}")
        cmd = [py, str(script), "--job", str(job_path)]
        log(f"image native {py} mode={mode}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(line, flush=True)
            if _log_shipper is not None:
                _log_shipper.add(line)
            if line.startswith("[image] progress "):
                try:
                    frac = float(line.split()[2])
                    on_progress(max(0.0, min(1.0, frac)))
                except (IndexError, ValueError):
                    pass
            if render_should_stop():
                proc.terminate()
                raise RuntimeError("Render aborted by operator")
    finally:
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"image job exited with code {proc.returncode}")
    files = sorted(out_dir.glob("image_*.png"))
    if not files:
        raise RuntimeError("image_failed: no PNG written")
    return files


DEFAULT_LORA_DOCKER = os.environ.get("LORA_DOCKER_IMAGE", "millicloud-lora:qwen")


def lora_train_script() -> Path:
    env = os.environ.get("LORA_TRAIN")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for p in (
        here.parent / "lora" / "train.py",
        here / "train.py",
        Path("/opt/lora/train.py"),
    ):
        if p.is_file():
            return p
    return here.parent / "lora" / "train.py"


def lora_infer_script() -> Path:
    env = os.environ.get("LORA_INFER")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for p in (here.parent / "lora" / "infer.py", here / "infer.py"):
        if p.is_file():
            return p
    return here.parent / "lora" / "infer.py"


def _post_lora_try(try_id: str, payload: dict) -> None:
    token = os.environ.get("RY_WORKER_TOKEN") or ""
    api("POST", f"/agent/tries/{try_id}/complete", token, json=payload).raise_for_status()


def run_lora_try(claim: dict) -> None:
    """Generate the base answer and the adapter answer for one prompt."""
    settings = claim.get("engine_settings") or {}
    try_id = str(settings.get("try_id") or "")
    if not try_id:
        raise RuntimeError("try_failed: claim is missing try_id")
    if sys.platform == "darwin" or not has_nvidia():
        err = "no_gpu: comparing an adapter runs on NVIDIA workers only"
        _post_lora_try(try_id, {"success": False, "error": err})
        raise RuntimeError(err)
    if not shutil.which("docker"):
        err = "no_libs: Docker is required. Build millicloud-lora:qwen on this host."
        _post_lora_try(try_id, {"success": False, "error": err})
        raise RuntimeError(err)
    script = lora_infer_script()
    if not script.is_file():
        err = "no_libs: lora/infer.py is missing next to the agent"
        _post_lora_try(try_id, {"success": False, "error": err})
        raise RuntimeError(err)

    if RENDER_MODE == "secure":
        workdir = SANDBOX_ROOT / "jobs" / uuid.uuid4().hex
        workdir.mkdir(parents=True, exist_ok=True)
    elif RENDER_MODE == "trusted":
        workdir = Path(tempfile.mkdtemp(prefix="ry_try_"))
    else:
        err = f"try_failed: unsupported RY_RENDER_MODE {RENDER_MODE}"
        _post_lora_try(try_id, {"success": False, "error": err})
        raise RuntimeError(err)
    try:
        adapter_dir = workdir / "adapter"
        out_dir = workdir / "out"
        adapter_dir.mkdir()
        out_dir.mkdir()
        download_scene(settings["adapter_url"], adapter_dir / "adapter_model.safetensors")
        download_scene(settings["adapter_config_url"], adapter_dir / "adapter_config.json")
        shutil.copy(script, workdir / "infer.py")
        job = {
            "prompt": settings.get("prompt") or "",
            "model_id": settings.get("model_id") or "",
            "model_revision": settings.get("model_revision") or "main",
            "adapter_dir": "/work/adapter",
            "output": "/work/out/try_result.json",
        }
        (workdir / "lora-try.json").write_text(json.dumps(job), encoding="utf-8")
        image = settings.get("docker_image") or DEFAULT_LORA_DOCKER
        cache = Path(os.environ.get("LORA_CACHE") or (Path.home() / ".cache" / "millicloud-lora"))
        cache.mkdir(parents=True, exist_ok=True)
        _reclaim_mount(cache, image)
        _reclaim_mount(workdir, image)
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--security-opt", "label=disable",
            "--entrypoint", "python3",
            "-v", f"{workdir.resolve()}:/work",
            "-v", f"{cache.resolve()}:/cache",
            "-e", "HF_HOME=/cache/hf",
            "-e", "HOME=/tmp",
            "-e", "PYTHONUNBUFFERED=1",
            image, "/work/infer.py", "--job", "/work/lora-try.json",
        ]
        log(f"lora compare docker {image}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(line, flush=True)
            if render_should_stop():
                proc.terminate()
                proc.wait()
                raise RuntimeError("Render aborted by operator")
        code = proc.wait()
        result_path = out_dir / "try_result.json"
        if code != 0 or not result_path.is_file():
            raise RuntimeError(f"try_failed: compare exited with code {code}")
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        _post_lora_try(try_id, {
            "success": True,
            "base_text": str(saved.get("base") or "")[:8000],
            "adapter_text": str(saved.get("adapter") or "")[:8000],
        })
        log("compare complete")
    except Exception as e:
        try:
            _post_lora_try(try_id, {"success": False, "error": str(e)[:1900]})
        except requests.RequestException:
            pass
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _zip_latest_checkpoint(ckpt_dir: Path, dest: Path) -> Path | None:
    """Pack the newest Trainer checkpoint-* directory. Skip if nothing was saved yet."""
    if not ckpt_dir.is_dir():
        return None
    found = sorted(p for p in ckpt_dir.glob("checkpoint-*") if p.is_dir())
    if not found:
        return None
    latest = found[-1]
    if dest.exists():
        dest.unlink()
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in latest.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(Path(latest.name) / path.relative_to(latest)))
    return dest


def _owned_by_user(path: Path) -> bool:
    uid = os.getuid()
    for dirpath, _dirnames, filenames in os.walk(path):
        if Path(dirpath).stat().st_uid != uid:
            return False
        for name in filenames:
            if (Path(dirpath) / name).stat().st_uid != uid:
                return False
    return True


def _reclaim_mount(path: Path, image: str) -> None:
    """A root container leaves root-owned files on the bind mount. The host user cannot read them."""
    if _owned_by_user(path):
        return
    log(f"reclaiming root-owned files under {path}")
    proc = subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "chown",
            "--security-opt", "label=disable",
            "-v", f"{path.resolve()}:/data",
            image,
            "-R", f"{os.getuid()}:{os.getgid()}", "/data",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"no_libs: could not reclaim {path}: {detail}")


def run_lora(claim: dict, workdir: Path, dataset: Path, on_progress, on_checkpoint) -> tuple[list[Path], dict]:
    """One QLoRA try. NVIDIA Docker only — Mac workers must not reach this."""
    if sys.platform == "darwin" or not has_nvidia():
        raise RuntimeError("no_gpu: LoRA training runs on NVIDIA workers only")
    if not shutil.which("docker"):
        raise RuntimeError(
            "no_libs: Docker is required. Build millicloud-lora:qwen on this host (see lora.md)."
        )
    if not dataset.is_file():
        raise RuntimeError("scene_error: LoRA dataset was not downloaded")

    settings = claim.get("engine_settings") or {}
    trial = settings.get("trial") or {}
    try_no = int(claim.get("frame_start") or 1)
    out_dir = workdir / "out"
    ckpt_dir = workdir / "ckpt"
    out_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    resume_for_job = None
    ckpt_url = claim.get("checkpoint_url")
    if ckpt_url:
        archive = workdir / "resume.zip"
        download_scene(ckpt_url, archive)
        resume_dir = workdir / "resume"
        resume_dir.mkdir(exist_ok=True)
        safe_extract(archive, resume_dir)
        resume_for_job = "/work/resume"
        log(f"lora resume archive for try {try_no}")

    job = {
        "dataset": "/work/" + dataset.name,
        "output_dir": "/work/out",
        "checkpoint_dir": "/work/ckpt",
        "model_id": settings.get("model_id") or "",
        "model_revision": settings.get("model_revision") or "main",
        "hardware": settings.get("hardware") or {},
        "dataset_spec": settings.get("dataset") or {},
        "recipe": settings.get("recipe") or {},
        "trial": {
            "rank": int(trial.get("rank") or settings.get("rank") or 16),
            "lr": float(trial.get("lr") or settings.get("lr") or 2e-4),
            "epochs": int(trial.get("epochs") or settings.get("epochs") or 1),
            "max_seq": int(trial.get("max_seq") or settings.get("max_seq") or 2048),
        },
    }
    if resume_for_job:
        job["resume"] = resume_for_job
    job_path = workdir / "lora-job.json"
    job_path.write_text(json.dumps(job, indent=2))

    script = lora_train_script()
    image = settings.get("docker_image") or DEFAULT_LORA_DOCKER
    cache = Path(os.environ.get("LORA_CACHE") or (Path.home() / ".cache" / "millicloud-lora"))
    cache.mkdir(parents=True, exist_ok=True)
    # The image has no USER, so Docker writes the adapter as root. On Fedora,
    # SELinux also labels that mount so the agent cannot open it. Run as the
    # host user and keep the directory label.
    _reclaim_mount(cache, image)
    _reclaim_mount(workdir, image)
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--security-opt", "label=disable",
        "-v", f"{workdir.resolve()}:/work",
        "-v", f"{cache.resolve()}:/cache",
        "-e", "HF_HOME=/cache/hf",
        "-e", "HOME=/tmp",
        "-e", "PYTHONUNBUFFERED=1",
    ]
    if script.is_file():
        cmd += ["-v", f"{script.resolve()}:/opt/lora/train.py:ro"]
    cmd += [image, "--job", "/work/lora-job.json"]
    log(f"lora docker {image} try={try_no} rank={job['trial']['rank']}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    metrics: dict = {}
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(line, flush=True)
            if _log_shipper is not None:
                _log_shipper.add(line)
            if line.startswith("[lora] progress "):
                try:
                    on_progress(max(0.0, min(1.0, float(line.split()[2]))))
                except (IndexError, ValueError):
                    pass
            elif line.startswith("[lora] val_loss "):
                try:
                    metrics["val_loss"] = float(line.split()[2])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("[lora] checkpoint "):
                packed = _zip_latest_checkpoint(ckpt_dir, workdir / f"checkpoint-t{try_no}.zip")
                if packed is not None:
                    try:
                        on_checkpoint(packed)
                    except Exception as e:
                        log(f"checkpoint upload failed (training continues): {e}")
            if render_should_stop():
                proc.terminate()
                raise RuntimeError("Render aborted by operator")
    finally:
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"lora job exited with code {proc.returncode}")

    result_path = out_dir / "train_result.json"
    if result_path.is_file():
        try:
            saved = json.loads(result_path.read_text())
            if "val_loss" in saved:
                metrics["val_loss"] = float(saved["val_loss"])
            if "global_step" in saved:
                metrics["global_step"] = int(saved["global_step"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    adapter = out_dir / "adapter_model.safetensors"
    config = out_dir / "adapter_config.json"
    if not adapter.is_file():
        raise RuntimeError("lora_failed: adapter_model.safetensors was not written")
    adapter_dest = out_dir / f"adapter-t{try_no}.safetensors"
    adapter.rename(adapter_dest)
    files = [adapter_dest]
    if config.is_file():
        config_dest = out_dir / f"adapter-t{try_no}.config.json"
        config.rename(config_dest)
        files.append(config_dest)
    if result_path.is_file():
        result_dest = out_dir / f"train-t{try_no}.json"
        result_path.rename(result_dest)
        files.append(result_dest)
    return files, metrics


GPU_ENABLE_EXPR = (
    "import bpy;"
    "p=bpy.context.preferences.addons['cycles'].preferences;"
    "p.compute_device_type=next(t for t in ('OPTIX','CUDA','HIP','METAL','NONE') "
    "  if t=='NONE' or any(d.type==t for d in (p.get_devices() or []) or p.devices));"
    "[setattr(d,'use',d.type!='CPU') for d in p.devices];"
    "bpy.context.scene.cycles.device='GPU' if p.compute_device_type!='NONE' else 'CPU';"
    "print('[gpu]',p.compute_device_type)"
)


def write_settings_script(job: dict, dest: Path) -> Path | None:
    """Write a Blender --python script. --python-expr cannot run multi-line overrides."""
    s = job.get("engine_settings") or {}
    if not isinstance(s, dict):
        return None
    clean = {k: s[k] for k in s if k != "chunk_size"}
    if not clean:
        return None
    dest.write_text(
        "import bpy, json\n"
        f"s = json.loads({json.dumps(json.dumps(clean))})\n"
        "def apply(sc):\n"
        "    r = sc.render\n"
        "    eng = str(s.get('engine') or '')\n"
        "    if eng == 'cycles':\n"
        "        r.engine = 'CYCLES'\n"
        "    elif eng == 'eevee':\n"
        "        for name in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):\n"
        "            try:\n"
        "                r.engine = name\n"
        "                break\n"
        "            except Exception:\n"
        "                pass\n"
        "    if s.get('resolution_x'): r.resolution_x = int(s['resolution_x'])\n"
        "    if s.get('resolution_y'): r.resolution_y = int(s['resolution_y'])\n"
        "    if s.get('resolution_pct') is not None:\n"
        "        r.resolution_percentage = int(s['resolution_pct'])\n"
        "    n = s.get('samples')\n"
        "    if n is not None:\n"
        "        n = int(n)\n"
        "        cy = getattr(sc, 'cycles', None)\n"
        "        if cy is not None:\n"
        "            cy.samples = n\n"
        "            if hasattr(cy, 'preview_samples'): cy.preview_samples = n\n"
        "            if hasattr(cy, 'use_adaptive_sampling'): cy.use_adaptive_sampling = False\n"
        "            if hasattr(cy, 'adaptive_min_samples'): cy.adaptive_min_samples = n\n"
        "        ee = getattr(sc, 'eevee', None)\n"
        "        if ee is not None:\n"
        "            for attr in ('taa_render_samples', 'taa_samples'):\n"
        "                if hasattr(ee, attr): setattr(ee, attr, n)\n"
        "        for vl in sc.view_layers:\n"
        "            if hasattr(vl, 'samples'): vl.samples = n\n"
        "            vlc = getattr(vl, 'cycles', None)\n"
        "            if vlc is not None:\n"
        "                if hasattr(vlc, 'use_layer_samples'):\n"
        "                    try: vlc.use_layer_samples = 'USE_RENDER'\n"
        "                    except Exception: pass\n"
        "                if hasattr(vlc, 'samples'): vlc.samples = n\n"
        "    if s.get('use_denoising') is not None and hasattr(sc, 'cycles'):\n"
        "        sc.cycles.use_denoising = bool(s['use_denoising'])\n"
        "    print('[settings]', sc.name, r.engine, r.resolution_x, 'x', r.resolution_y,\n"
        "          '@', r.resolution_percentage, '% samples=', getattr(getattr(sc,'cycles',None),'samples', '?'))\n"
        "for sc in bpy.data.scenes:\n"
        "    apply(sc)\n"
        "if bpy.context.scene:\n"
        "    apply(bpy.context.scene)\n"
    )
    return dest

SAVED_RE = re.compile(r"^Saved: '(.+?)'")


def render(blend: Path, out_dir: Path, job: dict, on_frame, event_sink=None) -> list[Path]:
    frame_start, frame_end = job["frame_start"], job["frame_end"]
    cmd = [
        BLENDER, "-b", str(blend),
        "--factory-startup", "--disable-autoexec",
        "--python-expr", GPU_ENABLE_EXPR,
    ]
    settings_py = write_settings_script(job, out_dir.parent / "_mc_settings.py")
    if settings_py:
        cmd += ["--python", str(settings_py)]
        log(f"applying engine_settings {job.get('engine_settings')}")
    cmd += [
        "-o", str(out_dir / "frame_####"),
        "-F", "OPEN_EXR", "-x", "1",
    ]
    if job.get("is_animation") or frame_end > frame_start:
        cmd += ["-s", str(frame_start), "-e", str(frame_end), "-a"]
    else:
        cmd += ["-f", str(frame_start)]

    log(f"rendering frames {frame_start}-{frame_end}")
    log(f"blender cmd: {cmd[0]} … {blend.name}")
    saved: list[Path] = []
    blender_tail: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=render_env(out_dir.parent),
    )
    timeout_fired = threading.Event()

    def kill_on_timeout() -> None:
        timeout_fired.set()
        try:
            proc.kill()
        except OSError:
            pass

    timer = threading.Timer(RENDER_TIMEOUT_SECONDS, kill_on_timeout)
    timer.daemon = True
    timer.start()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                blender_tail.append(line)
                if len(blender_tail) > 80:
                    blender_tail = blender_tail[-60:]
                emit_blender(line, event_sink)
            m = SAVED_RE.match(line)
            if m:
                saved.append(Path(m.group(1)))
                on_frame(len(saved))
                log(f"frame saved ({len(saved)}/{frame_end - frame_start + 1})")
            if render_should_stop():
                proc.terminate()
                if _abort_render.is_set() and not _stop.is_set():
                    raise RuntimeError("Render aborted by operator")
                raise RuntimeError("Agent stopped by signal")
    finally:
        timer.cancel()
    proc.wait()
    if timeout_fired.is_set():
        raise RuntimeError(f"Blender exceeded render timeout ({RENDER_TIMEOUT_SECONDS}s)")
    if proc.returncode != 0:
        code = proc.returncode
        sig = f" (signal {-code})" if code is not None and code < 0 else ""
        tail = "\n".join(blender_tail[-30:]) or "(no Blender output captured)"
        log(f"blender crash{sig}. last output:\n{tail}")
        raise RuntimeError(f"Blender exited with code {code}{sig}. Last output:\n{tail}")
    if not saved:
        raise RuntimeError("Blender produced no frames")
    return saved


def exr_to_png(exr_files: list[Path], event_sink=None) -> list[Path]:
    script = (
        "import bpy,sys\n"
        "files=sys.argv[sys.argv.index('--')+1:]\n"
        "s=bpy.context.scene; s.render.image_settings.file_format='PNG'\n"
        "for f in files:\n"
        "    img=bpy.data.images.load(f)\n"
        "    out=f[:-4]+'.png'\n"
        "    img.save_render(out)\n"
        "    print('CONVERTED:'+out)\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        res = subprocess.run(
            [
                BLENDER, "-b", "--factory-startup", "--disable-autoexec",
                "--python", script_path, "--", *[str(p) for p in exr_files],
            ],
            capture_output=True, text=True, timeout=3600,
            env=render_env(exr_files[0].parent.parent if exr_files else Path.cwd()),
        )
        if _log_shipper is not None:
            for line in (res.stdout or "").splitlines():
                if line.strip():
                    _log_shipper.add(line)
                    if event_sink is not None:
                        event_sink({"type": "log", "text": line})
        pngs = [
            Path(l.split("CONVERTED:", 1)[1])
            for l in res.stdout.splitlines() if l.startswith("CONVERTED:")
        ]
        if len(pngs) != len(exr_files):
            raise RuntimeError("PNG conversion incomplete")
        return pngs
    finally:
        os.unlink(script_path)


def _safe_sandbox_path(relative: str) -> Path:
    """Resolve a renderer path and reject traversal outside the shared volume."""
    if not relative or Path(relative).is_absolute():
        raise RuntimeError("Renderer path must be relative")
    path = (SANDBOX_ROOT / relative).resolve()
    root = SANDBOX_ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Unsafe renderer path: {relative}")
    return path


def renderer_available() -> bool:
    try:
        ready = SANDBOX_ROOT / "renderer.ready"
        return ready.exists() and time.time() - ready.stat().st_mtime < 30
    except OSError:
        return False


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _renderer_event(path: Path, event: dict) -> None:
    event = {"t": time.time(), **event}
    with open(path, "a") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")
        f.flush()


def render_via_sidecar(blend: Path, out_dir: Path, workdir: Path, job: dict, on_frame) -> list[Path]:
    """Submit a render to the no-network renderer sidecar over the shared volume."""
    root = SANDBOX_ROOT.resolve()
    workdir = workdir.resolve()
    try:
        workdir.relative_to(root)
        blend.relative_to(root)
        out_dir.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Secure render workdir must be inside RY_SANDBOX_ROOT") from exc

    inbox = root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    ready = root / "renderer.ready"
    ready_deadline = time.monotonic() + 30
    while not renderer_available() and time.monotonic() < ready_deadline and not _stop.is_set():
        _stop.wait(0.5)
    if not renderer_available():
        raise RuntimeError("Secure renderer sidecar is not ready")
    request_id = uuid.uuid4().hex
    events_path = workdir / "renderer-events.ndjson"
    result_path = workdir / "renderer-result.json"
    request_path = inbox / f"{request_id}.json"
    manifest = {
        "request_id": request_id,
        "workdir": str(workdir.relative_to(root)),
        "blend": str(blend.relative_to(root)),
        "output_dir": str(out_dir.relative_to(root)),
        "frame_start": job["frame_start"],
        "frame_end": job["frame_end"],
        "is_animation": bool(job.get("is_animation")),
        "output_format": job["output_format"],
        "engine_settings": job.get("engine_settings") or {},
        "result_path": str(result_path.relative_to(root)),
        "events_path": str(events_path.relative_to(root)),
    }
    _write_json_atomic(request_path, manifest)
    log(f"submitted secure render to renderer sidecar ({request_id[:10]})")

    seen_events = 0
    started = time.monotonic()
    while not _stop.is_set():
        if time.monotonic() - started > RENDER_TIMEOUT_SECONDS:
            raise RuntimeError(f"Secure renderer exceeded timeout ({RENDER_TIMEOUT_SECONDS}s)")
        if events_path.exists():
            try:
                lines = events_path.read_text().splitlines()
            except OSError:
                lines = []
            for raw in lines[seen_events:]:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "log" and event.get("text"):
                    log(f"[renderer] {event['text']}")
                elif event.get("type") == "frame":
                    on_frame(int(event.get("frames", 0)))
                elif event.get("type") == "metrics" and isinstance(event.get("metrics"), dict):
                    _runtime_metrics.update(event["metrics"])
            seen_events = len(lines)
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                result = None
            if result is not None:
                if not result.get("success"):
                    raise RuntimeError(result.get("error") or "Secure renderer failed")
                outputs: list[Path] = []
                for name in result.get("outputs", []):
                    output = _safe_sandbox_path(name)
                    if output.parent != out_dir.resolve() or not output.is_file():
                        raise RuntimeError(f"Renderer output missing or outside output directory: {name}")
                    outputs.append(output)
                if not outputs:
                    raise RuntimeError("Secure renderer produced no frames")
                return outputs
        _stop.wait(0.5)
    raise RuntimeError("Agent stopped while waiting for secure renderer")


def renderer_run_request(manifest: dict) -> None:
    """Run one fixed render request as the networkless sidecar process."""
    workdir = _safe_sandbox_path(manifest["workdir"])
    blend = _safe_sandbox_path(manifest["blend"])
    out_dir = _safe_sandbox_path(manifest["output_dir"])
    result_path = _safe_sandbox_path(manifest["result_path"])
    events_path = _safe_sandbox_path(manifest["events_path"])
    if blend.suffix.lower() != ".blend" or not blend.is_file():
        raise RuntimeError("Renderer scene is missing or is not a .blend file")
    if out_dir.parent != workdir:
        raise RuntimeError("Renderer output directory must be directly under workdir")
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    _renderer_event(events_path, {"type": "metrics", "metrics": collect_runtime_metrics()})
    _renderer_event(events_path, {"type": "log", "text": f"rendering frames {manifest['frame_start']}-{manifest['frame_end']}"})
    job = {
        "frame_start": int(manifest["frame_start"]),
        "frame_end": int(manifest["frame_end"]),
        "is_animation": bool(manifest.get("is_animation")),
        "output_format": manifest["output_format"],
        "engine_settings": manifest.get("engine_settings") or {},
    }
    exrs = render(
        blend,
        out_dir,
        job,
        on_frame=lambda n: _renderer_event(events_path, {"type": "frame", "frames": n}),
        event_sink=lambda event: _renderer_event(events_path, event),
    )
    to_upload: list[Path] = []
    if job["output_format"] in ("exr", "both"):
        to_upload += exrs
    if job["output_format"] in ("png", "both"):
        _renderer_event(events_path, {"type": "log", "text": "converting EXR -> PNG"})
        to_upload += exr_to_png(exrs, event_sink=lambda event: _renderer_event(events_path, event))
    outputs = [str(path.resolve().relative_to(SANDBOX_ROOT.resolve())) for path in to_upload]
    _write_json_atomic(result_path, {"success": True, "outputs": outputs})
    _renderer_event(events_path, {"type": "log", "text": "renderer complete"})


def renderer_loop() -> int:
    """Process manifests from the shared volume; this process never contacts the API."""
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    inbox = SANDBOX_ROOT / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    ready = SANDBOX_ROOT / "renderer.ready"
    ready.write_text(json.dumps({"agent_version": AGENT_VERSION, "started_at": time.time()}))
    print("[renderer] secure sidecar ready", flush=True)
    try:
        while not _stop.is_set():
            for request_path in sorted(inbox.glob("*.json")):
                running_path = request_path.with_suffix(".running")
                manifest: dict = {}
                try:
                    request_path.rename(running_path)
                    manifest = json.loads(running_path.read_text())
                    try:
                        renderer_run_request(manifest)
                    except Exception as exc:
                        result_path = _safe_sandbox_path(manifest["result_path"])
                        _write_json_atomic(result_path, {"success": False, "error": str(exc)[:1900]})
                    finally:
                        running_path.unlink(missing_ok=True)
                except Exception as exc:
                    print(f"[renderer] invalid request: {exc}", flush=True)
                    if manifest.get("result_path"):
                        try:
                            _write_json_atomic(
                                _safe_sandbox_path(manifest["result_path"]),
                                {"success": False, "error": str(exc)[:1900]},
                            )
                        except Exception:
                            pass
            _stop.wait(0.5)
            try:
                ready.touch()
            except OSError:
                pass
    finally:
        ready.unlink(missing_ok=True)
    return 0


def heartbeat_loop(job_id: str, token: str, state: dict, total_frames: int, chunk_id: str | None) -> None:
    while not state["done"] and not render_should_stop():
        try:
            payload = {
                "frames_done": state["frames"],
                "progress": state.get(
                    "progress",
                    min(0.99, state["frames"] / max(1, total_frames)),
                ),
                "gpu_name": state.get("gpu"),
            }
            if chunk_id:
                payload["chunk_id"] = chunk_id
            r = api("POST", f"/agent/jobs/{job_id}/heartbeat", token, json=payload)
            if r.status_code == 409:
                abort_current_render("lease lost or chunk requeued")
                return
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    data = {}
                if data.get("command") == "abort_current":
                    abort_current_render("server command")
                    return
        except requests.RequestException as e:
            log(f"heartbeat failed (will retry): {e}")
        _stop.wait(HEARTBEAT_INTERVAL)


def upload_outputs(job_id: str, token: str, files: list[Path]) -> None:
    for f in files:
        size = f.stat().st_size
        frame = None
        m = re.search(r"frame_(\d+)", f.name)
        if m:
            frame = int(m.group(1))
        r = api(
            "POST", f"/agent/jobs/{job_id}/outputs", token,
            json={"filename": f.name, "size": size, "frame": frame},
        )
        r.raise_for_status()
        put_url = r.json()["upload_url"]
        with open(f, "rb") as fh:
            # Adapters and checkpoints are larger than a PNG.
            timeout = 3600 if size > 64 * 1024**2 else 600
            up = requests.put(put_url, data=fh, headers={"Content-Length": str(size)}, timeout=timeout)
            up.raise_for_status()
        log(f"uploaded {f.name} ({size} bytes)")


def run_job(claim: dict, token: str) -> None:
    global _log_shipper, _run_policy, _runtime_metrics
    job_id = str(claim["job_id"])
    chunk_id = str(claim["chunk_id"]) if claim.get("chunk_id") else None
    if claim.get("run_policy"):
        _run_policy = claim["run_policy"]
    total = claim["frame_end"] - claim["frame_start"] + 1
    state = {"frames": 0, "done": False, "gpu": detect_gpu()}
    _status.update({
        "presence": "busy",
        "job_id": job_id,
        "chunk_id": chunk_id,
        "frames_done": 0,
        "last_error": None,
        "mode": RENDER_MODE,
    })
    t0 = time.time()

    shipper = LogShipper(job_id, token, chunk_id=chunk_id)
    _log_shipper = shipper
    shipper.start()
    if _worker_log_shipper is not None:
        _worker_log_shipper.set_context(job_id, chunk_id)
        _worker_log_shipper.add(
            f"[agent] rendering job={job_id} chunk={chunk_id or 'oneshot'} "
            f"frames={claim['frame_start']}-{claim['frame_end']}"
        )
    shipper.add(f"[agent] start job={job_id} chunk={chunk_id or 'oneshot'} frames={claim['frame_start']}-{claim['frame_end']}")
    if state["gpu"]:
        shipper.add(f"[agent] gpu: {state['gpu']}")

    hb = threading.Thread(
        target=heartbeat_loop,
        args=(job_id, token, state, total, chunk_id),
        daemon=True,
    )
    hb.start()

    if RENDER_MODE == "secure":
        workdir = SANDBOX_ROOT / "jobs" / uuid.uuid4().hex
        workdir.mkdir(parents=True, exist_ok=True)
    elif RENDER_MODE == "trusted":
        workdir = Path(tempfile.mkdtemp(prefix="ry_"))
    else:
        raise RuntimeError(f"Unsupported RY_RENDER_MODE: {RENDER_MODE}")
    try:
        # Free disk check (soft)
        free = shutil.disk_usage(workdir).free
        if free < 512 * 1024**2:
            raise RuntimeError(f"No space left on device (free={free} bytes)")

        engine = (claim.get("engine") or "blender").lower()
        settings = claim.get("engine_settings") or {}
        scene = workdir / (claim.get("scene_filename") or "scene.bin")
        need_scene = engine != "image" or str(settings.get("mode") or "") == "img2img"
        if need_scene:
            download_scene(claim["scene_url"], scene)

        if engine == "image":
            init = scene if str(settings.get("mode") or "") == "img2img" else None

            def _img_progress(p: float) -> None:
                state["progress"] = p
                state["frames"] = max(state["frames"], int(p * 100))

            to_upload = run_image(claim, workdir, init, on_progress=_img_progress)
        elif engine == "lora":
            def _lora_progress(p: float) -> None:
                state["progress"] = p
                state["frames"] = max(state["frames"], int(p * 100))

            def _lora_checkpoint(path: Path) -> None:
                upload_outputs(job_id, token, [path])

            to_upload, lora_metrics = run_lora(
                claim, workdir, scene, on_progress=_lora_progress, on_checkpoint=_lora_checkpoint,
            )
            if lora_metrics:
                state["metrics"] = lora_metrics
        elif engine == "openfoam":
            if scene.suffix != ".zip":
                raise RuntimeError("OpenFOAM jobs must be a .zip of the case directory")
            extract_dir = workdir / "scene"
            extract_dir.mkdir()
            safe_extract(scene, extract_dir)
            case = find_foam_case(extract_dir)
            log(f"using OpenFOAM case: {case}")

            def _of_progress(p: float) -> None:
                state["progress"] = p
                state["frames"] = max(state["frames"], int(p * 100))

            to_upload = run_openfoam(case, claim, on_progress=_of_progress)
        else:
            if scene.suffix == ".zip":
                extract_dir = workdir / "scene"
                extract_dir.mkdir()
                safe_extract(scene, extract_dir)
                blend = find_blend(extract_dir)
                log(f"using blend: {blend}")
            else:
                blend = scene

            out_dir = workdir / "out"
            out_dir.mkdir()
            if RENDER_MODE == "secure":
                to_upload = render_via_sidecar(
                    blend,
                    out_dir,
                    workdir,
                    claim,
                    on_frame=lambda n: state.update(frames=n),
                )
            else:
                to_upload = render(
                    blend, out_dir, claim, on_frame=lambda n: state.update(frames=n)
                )

            fmt = claim["output_format"]
            packed: list[Path] = []
            if fmt in ("exr", "both"):
                packed += to_upload
            if fmt in ("png", "both"):
                log("converting EXR -> PNG (view transform applied)")
                if RENDER_MODE == "secure":
                    packed = to_upload
                else:
                    packed += exr_to_png(to_upload)
            to_upload = packed

        upload_outputs(job_id, token, to_upload)
        state["done"] = True
        _status.update({"presence": "idle", "frames_done": total, "job_id": None, "chunk_id": None})
        body = {
            "success": True,
            "frames_done": total,
            "render_seconds": int(time.time() - t0),
        }
        if chunk_id:
            body["chunk_id"] = chunk_id
        if state.get("metrics"):
            body["metrics"] = state["metrics"]
        log("chunk/job complete ✔")
        # Flush shipper before complete so the last lines are not 409'd.
        shipper.close()
        _log_shipper = None
        api("POST", f"/agent/jobs/{job_id}/complete", token, json=body).raise_for_status()
    except Exception as e:
        state["done"] = True
        code = classify_error(e)
        _status.update({"presence": "error", "last_error": str(e)[:1900]})
        log(f"job failed [{code}]: {e}")
        try:
            shipper.close()
        except Exception:
            pass
        _log_shipper = None
        try:
            body = {
                "success": False,
                "error": str(e)[:1900],
                "error_code": code,
                "render_seconds": int(time.time() - t0),
            }
            if chunk_id:
                body["chunk_id"] = chunk_id
            api("POST", f"/agent/jobs/{job_id}/complete", token, json=body)
        except requests.RequestException:
            pass
        raise
    finally:
        if _log_shipper is not None:
            shipper.close()
            _log_shipper = None
        if _worker_log_shipper is not None:
            _worker_log_shipper.set_context(None, None)
        shutil.rmtree(workdir, ignore_errors=True)


def worker_heartbeat_loop(wtoken: str, state: dict) -> None:
    """Keeps last_seen + specs fresh; pulls run_policy instructions from the API."""
    global _run_policy, _runtime_metrics
    while not _stop.is_set():
        try:
            specs = collect_specs()
            _runtime_metrics = collect_runtime_metrics()
            specs["runtime_metrics"] = dict(_runtime_metrics)
            specs["render_mode"] = RENDER_MODE
            specs["sandbox"] = {
                "mode": RENDER_MODE,
                "sidecar_required": RENDER_MODE == "secure",
                "available": RENDER_MODE == "trusted" or (SANDBOX_ROOT / "renderer.ready").exists(),
                "network_isolated": RENDER_MODE == "secure",
                "token_isolated": RENDER_MODE == "secure",
            }
            presence = state.get("presence", "idle")
            _status.update({
                "presence": presence,
                "job_id": state.get("job_id"),
                "chunk_id": state.get("chunk_id"),
                "metrics": dict(_runtime_metrics),
            })
            body = {
                "presence": presence,
                "hostname": specs.get("hostname") or socket.gethostname(),
                "gpu_name": specs.get("gpu_name") or detect_gpu(),
                "agent_version": AGENT_VERSION,
                "specs": specs,
                "current_job_id": state.get("job_id"),
                "current_chunk_id": state.get("chunk_id"),
            }
            mid = os.environ.get("RY_MACHINE_ID")
            if mid:
                body["machine_id"] = mid
            sid = os.environ.get("RY_SESSION_ID")
            if sid:
                body["session_id"] = sid
            if state.get("error"):
                body["error"] = state["error"]
                body["error_code"] = state.get("error_code")
                body["presence"] = "error"
            r = api("POST", "/agent/worker/heartbeat", wtoken, json=body)
            if r.status_code == 401:
                log("worker token revoked — exiting")
                _stop.set()
                return
            if r.status_code == 200:
                data = r.json()
                _run_policy = data.get("run_policy") or _run_policy
                if data.get("command") == "abort_current":
                    abort_current_render("server command")
                if data.get("render_mode") and data["render_mode"] != RENDER_MODE:
                    log(
                        f"render mode mismatch: API requires {data['render_mode']}, "
                        f"agent is configured for {RENDER_MODE}; restart with the matching mode"
                    )
                if data.get("revoked"):
                    log("worker revoked — exiting")
                    _stop.set()
                    return
                # Clear sticky error after successful report
                if presence != "error":
                    state.pop("error", None)
                    state.pop("error_code", None)
        except requests.RequestException as e:
            log(f"worker heartbeat failed: {e}")
        _stop.wait(WORKER_HB_INTERVAL)


def main() -> int:
    global _run_policy
    if RENDERER_MODE:
        return renderer_loop()
    if RENDER_MODE not in ("secure", "trusted"):
        print(f"[agent] RY_RENDER_MODE must be secure or trusted, got {RENDER_MODE!r}", flush=True)
        return 2
    if not API:
        print("[agent] RY_API is required", flush=True)
        return 2

    if os.environ.get("WORKER_MODE"):
        wtoken = os.environ.get("RY_WORKER_TOKEN")
        if not wtoken:
            print("[agent] WORKER_MODE requires RY_WORKER_TOKEN", flush=True)
            return 2

        start_control_panel()

        global _worker_log_shipper
        state: dict = {"presence": "idle"}
        _status["mode"] = RENDER_MODE
        _worker_log_shipper = WorkerLogShipper(wtoken)
        _worker_log_shipper.start()
        threading.Thread(target=worker_heartbeat_loop, args=(wtoken, state), daemon=True).start()
        log(f"worker mode v{AGENT_VERSION}: polling every {POLL_INTERVAL}s")

        while not _stop.is_set():
            if RENDER_MODE == "secure" and not renderer_available():
                state["presence"] = "error"
                state["error"] = "Secure renderer sidecar is not ready"
                state["error_code"] = "sandbox_unavailable"
                _status.update({"presence": "error", "last_error": state["error"]})
                _stop.wait(POLL_INTERVAL)
                continue
            if state.get("error_code") == "sandbox_unavailable":
                state.pop("error", None)
                state.pop("error_code", None)
                state["presence"] = "idle"
            # Drain / exit policies
            if _run_policy == "exit_after_job" and state.get("presence") != "busy":
                log("run_policy=exit_after_job — shutting down process")
                _worker_log_shipper.close()
                return 0
            if _run_policy == "drain":
                if state.get("presence") != "draining":
                    log("run_policy=drain — not accepting new chunks")
                state["presence"] = "draining"
                _stop.wait(POLL_INTERVAL)
                continue

            try:
                r = api("POST", "/agent/poll?supports=lora_try", wtoken)
                if r.status_code == 200 and r.json():
                    claim = r.json()
                    cid = str(claim["chunk_id"]) if claim.get("chunk_id") else None
                    if cid and cid == state.get("skip_chunk_id"):
                        log("skipping just-aborted chunk; waiting for another")
                        _stop.wait(POLL_INTERVAL)
                        continue
                    if claim.get("run_policy"):
                        _run_policy = claim["run_policy"]
                    state["presence"] = "busy"
                    state["job_id"] = str(claim["job_id"])
                    state["chunk_id"] = str(claim["chunk_id"]) if claim.get("chunk_id") else None
                    if claim.get("engine") == "lora_try":
                        log(f"claimed compare job {state['job_id']}")
                    else:
                        log(f"claimed chunk {state['chunk_id']} job {state['job_id']} "
                            f"frames {claim['frame_start']}-{claim['frame_end']}")
                    try:
                        _abort_render.clear()
                        if claim.get("engine") == "lora_try":
                            run_lora_try(claim)
                        else:
                            run_job(claim, claim["job_token"])
                        state["presence"] = "idle"
                        _status["presence"] = "idle"
                        log("chunk finished — back to idle")
                    except Exception as e:
                        code = classify_error(e)
                        if code == "cancelled":
                            state["skip_chunk_id"] = state.get("chunk_id")
                            state["presence"] = "idle"
                            _status["presence"] = "idle"
                            log("chunk aborted — in-progress frames discarded")
                        else:
                            state["presence"] = "error"
                            _status["presence"] = "error"
                            state["error"] = str(e)[:1900]
                            state["error_code"] = code
                            log(f"chunk failed [{code}]: {e}")
                    finally:
                        state["job_id"] = None
                        state["chunk_id"] = None
                        _status["job_id"] = None
                        _status["chunk_id"] = None
                        if _run_policy == "exit_after_job":
                            log("run_policy=exit_after_job — exiting after job")
                            _worker_log_shipper.close()
                            return 0
                        if _run_policy == "drain":
                            state["presence"] = "draining"
                        else:
                            state["presence"] = "idle"
                elif r.status_code == 401:
                    log("worker token invalid or revoked — exiting")
                    _worker_log_shipper.close()
                    return 1
                else:
                    state["presence"] = "draining" if _run_policy == "drain" else "idle"
            except requests.RequestException as e:
                log(f"poll failed (will retry): {e}")
            _stop.wait(POLL_INTERVAL)
        _worker_log_shipper.close()
        return 0

    job_id, token = os.environ.get("RY_JOB"), os.environ.get("RY_TOKEN")
    if not (job_id and token):
        print("[agent] one-shot mode requires RY_JOB and RY_TOKEN", flush=True)
        return 2
    start_control_panel()
    r = api("POST", f"/agent/jobs/{job_id}/start", token)
    if r.status_code != 200:
        print(f"[agent] could not start job: {r.status_code} {r.text}", flush=True)
        return 1
    run_job(r.json(), token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
