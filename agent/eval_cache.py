"""Resident evaluation model for a trusted host.

One container stays up for HOLD_SECONDS after the last prompt. A training claim
or an Unload command stops it before anything else uses the card. A MilliCloud
host never starts it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

HOLD_SECONDS = int(os.environ.get("EVAL_IDLE_SECONDS", "600"))
CONTAINER = "millicloud-eval"


class EvalCache:
    def __init__(self, log) -> None:
        self.log = log
        self.provider = "user"
        self.render_mode = os.environ.get("RY_RENDER_MODE", "secure").lower()
        self._lock = threading.Lock()

    def allowed(self) -> bool:
        if os.environ.get("EVAL_CACHE", "1") == "0":
            return False
        if self.provider == "millicloud":
            return False
        return self.render_mode == "trusted"

    def root(self) -> Path:
        cache = Path(os.environ.get("LORA_CACHE") or (Path.home() / ".cache" / "millicloud-lora"))
        return cache / "eval-live"

    def status(self) -> dict | None:
        if not self._running():
            return None
        meta = self.root() / "meta.json"
        if not meta.is_file():
            return {"state": "loading", "hold_seconds": HOLD_SECONDS, "host": "trusted"}
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"state": "loading", "hold_seconds": HOLD_SECONDS, "host": "trusted"}
        if not isinstance(data, dict):
            return None
        data["host"] = "trusted"
        data["hold_seconds"] = int(data.get("hold_seconds") or HOLD_SECONDS)
        return data

    def unload(self, reason: str) -> None:
        with self._lock:
            self._stop(reason)

    def _stop(self, reason: str) -> None:
        if not self._running() and not (self.root() / "meta.json").is_file():
            return
        self.log(f"unloading cached model ({reason})")
        root = self.root()
        root.mkdir(parents=True, exist_ok=True)
        (root / "unload").write_text("1", encoding="utf-8")
        subprocess.run(["docker", "stop", "-t", "30", CONTAINER], capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, text=True)
        for name in ("unload", "meta.json", "session.json"):
            (root / name).unlink(missing_ok=True)

    def same_adapter(self, claim: dict) -> bool:
        """True when this job's adapter is already the one loaded in the eval container."""
        if not self.allowed() or not self._running():
            return False
        current = self.root() / "session.json"
        if not current.is_file():
            return False
        try:
            have = json.loads(current.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        settings = claim.get("engine_settings") or {}
        want = self._signature(claim, settings, weight_bytes=have.get("adapter_bytes"))
        if not have.get("job_id"):
            return False
        if int(settings.get("adapter_bytes") or 0) and int(have.get("source_bytes") or 0):
            if int(have["source_bytes"]) != int(settings["adapter_bytes"]):
                return False
        return (
            have.get("model_id") == want["model_id"]
            and have.get("model_revision") == want["model_revision"]
            and have.get("adapter_label") == want["adapter_label"]
            and have.get("job_id") == want["job_id"]
        )

    def answer_loaded(self, claim: dict) -> dict:
        settings = claim.get("engine_settings") or {}
        try_id = str(settings.get("try_id") or "")
        label = str(settings.get("adapter_label") or "")
        with self._lock:
            if not self.same_adapter(claim):
                raise RuntimeError("try_failed: cached adapter changed before the prompt was sent")
            self._write_prompt(try_id, settings.get("prompt") or "")
        self.log(f"evaluation using cached adapter {label}")
        return self._wait_result(try_id)

    def answer(self, claim: dict, weight: Path, config: Path, *, reclaim) -> dict:
        settings = claim.get("engine_settings") or {}
        label = str(settings.get("adapter_label") or weight.name)
        image = str(settings.get("docker_image") or os.environ.get("LORA_DOCKER_IMAGE") or "millicloud-lora:latest")
        try_id = str(settings.get("try_id") or "")
        signature = self._signature(claim, settings, weight_bytes=weight.stat().st_size)
        with self._lock:
            self._ensure(signature, weight, config, image, reclaim)
            self._write_prompt(try_id, settings.get("prompt") or "")
        self.log(f"evaluation using cached model {signature['model_id']} ({label})")
        return self._wait_result(try_id)

    def _signature(self, claim: dict, settings: dict, weight_bytes) -> dict:
        return {
            "job_id": str(claim.get("job_id") or ""),
            "model_id": str(settings.get("model_id") or ""),
            "model_revision": str(settings.get("model_revision") or "main"),
            "adapter_label": str(settings.get("adapter_label") or ""),
            "adapter_bytes": int(weight_bytes or 0),
            "source_bytes": int(settings.get("adapter_bytes") or 0),
            "adapter_dir": "/session/adapter",
        }

    def _write_prompt(self, try_id: str, prompt: str) -> None:
        inbox = self.root() / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / f"{try_id}.json").write_text(
            json.dumps({"prompt": prompt}),
            encoding="utf-8",
        )

    def _ensure(self, signature: dict, weight: Path, config: Path, image: str, reclaim) -> None:
        root = self.root()
        current = root / "session.json"
        if self._running() and current.is_file():
            try:
                have = json.loads(current.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                have = {}
            if (
                have.get("model_id") == signature["model_id"]
                and have.get("model_revision") == signature["model_revision"]
                and have.get("adapter_label") == signature["adapter_label"]
                and have.get("job_id") == signature["job_id"]
                and have.get("adapter_bytes") == signature["adapter_bytes"]
            ):
                return
            self._stop("different model")
        elif self._running():
            self._stop("restart cache")
        root.mkdir(parents=True, exist_ok=True)
        for stale in ("unload", "meta.json"):
            (root / stale).unlink(missing_ok=True)
        adapter = root / "adapter"
        if adapter.exists():
            shutil.rmtree(adapter)
        adapter.mkdir(parents=True)
        (root / "inbox").mkdir(exist_ok=True)
        (root / "out").mkdir(exist_ok=True)
        shutil.copy(weight, adapter / "adapter_model.safetensors")
        shutil.copy(config, adapter / "adapter_config.json")
        here = Path(__file__).resolve().parent
        for name in ("infer.py", "serve.py"):
            src = here.parent / "lora" / name
            if not src.is_file():
                raise RuntimeError(f"no_libs: {src} is missing next to the agent")
            shutil.copy(src, root / name)
        current.write_text(json.dumps(signature), encoding="utf-8")
        reclaim(root.parent, image)
        reclaim(root, image)
        uid = f"{os.getuid()}:{os.getgid()}"
        cmd = [
            "docker", "run", "-d", "--rm", "--name", CONTAINER, "--gpus", "all",
            "--user", uid,
            "--security-opt", "label=disable",
            "--entrypoint", "python3",
            "-v", f"{root.resolve()}:/session",
            "-v", f"{root.parent.resolve()}:/cache",
            "-e", "HF_HOME=/cache/hf",
            "-e", "HOME=/tmp",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", f"EVAL_IDLE_SECONDS={HOLD_SECONDS}",
            image, "/session/serve.py",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"try_failed: could not start the evaluation cache: {detail}")
        self.log(f"cache starting {signature['model_id']} as {signature['adapter_label']} for {HOLD_SECONDS}s")
        threading.Thread(target=self._follow_logs, name="eval-cache-logs", daemon=True).start()

    def _follow_logs(self) -> None:
        proc = subprocess.Popen(
            ["docker", "logs", "-f", CONTAINER],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)

    def _wait_result(self, try_id: str) -> dict:
        out = self.root() / "out" / f"{try_id}.json"
        deadline = time.time() + 20 * 60
        while time.time() < deadline:
            if out.is_file():
                try:
                    data = json.loads(out.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"try_failed: bad cache result: {exc}") from exc
                out.unlink(missing_ok=True)
                if data.get("error") and not data.get("base"):
                    raise RuntimeError(f"try_failed: {data['error']}")
                return data
            if not self._running():
                raise RuntimeError("try_failed: cached model exited before the answer")
            time.sleep(0.5)
        raise RuntimeError("try_failed: cached model did not answer within 20 minutes")

    def _running(self) -> bool:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "true"
