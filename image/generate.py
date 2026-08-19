#!/usr/bin/env python3
"""MilliCloud image worker — FLUX.2 Klein 4B (Apache 2.0).

Reads a job JSON, preflights GPU/disk, generates PNG(s), prints [image] lines.
Apache-2.0 model: black-forest-labs/FLUX.2-klein-4B
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


KLEIN_ID = "black-forest-labs/FLUX.2-klein-4B"
SDXL_TURBO_ID = "stabilityai/sdxl-turbo"
DEFAULT_MODEL = "auto"
MIN_DISK_FIRST_GB = 8
MIN_DISK_CACHED_GB = 2
MIN_VRAM_MB = 2500
MIN_UNIFIED_GB = 8
# Klein 4B needs ~12 GB class cards at 1024. 8 GB (3070 Ti) uses SDXL Turbo.
KLEIN_MIN_TOTAL_MB = 10000


def log(msg: str) -> None:
    print(f"[image] {msg}", flush=True)


def fail(code: str, msg: str) -> None:
    log(f"ERROR [{code}] {msg}")
    raise SystemExit(f"{code}: {msg}")


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        fail("no_libs", "PyTorch is not installed. See image.md (setup).")
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _nvidia_query(field: str) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return max(int(x.strip()) for x in out.stdout.splitlines() if x.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def nvidia_free_vram_mb() -> int | None:
    return _nvidia_query("memory.free")


def nvidia_total_vram_mb() -> int | None:
    return _nvidia_query("memory.total")


def pick_model_id(requested: str, device: str) -> tuple[str, str]:
    """Return (hf_id, family) — family is klein | sdxl."""
    req = (requested or "auto").strip()
    if req in ("sdxl-turbo", SDXL_TURBO_ID):
        return SDXL_TURBO_ID, "sdxl"
    if req in ("flux2-klein-4b", KLEIN_ID):
        return KLEIN_ID, "klein"
    total = nvidia_total_vram_mb() if device == "cuda" else None
    if device == "cuda" and total is not None and total < KLEIN_MIN_TOTAL_MB:
        log(f"auto: GPU is {total} MB total — using SDXL Turbo (Klein needs ~12 GB class)")
        return SDXL_TURBO_ID, "sdxl"
    if device == "mps":
        ram = unified_ram_gb()
        if ram is not None and ram < 24:
            log(f"auto: {ram:.0f} GB unified — using SDXL Turbo")
            return SDXL_TURBO_ID, "sdxl"
    return KLEIN_ID, "klein"


def unified_ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return int(out.stdout.strip()) / (1024 ** 3)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return None


def hf_cache_dir() -> Path:
    raw = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "huggingface"


def model_cached(model_id: str) -> bool:
    cache = hf_cache_dir() / "hub"
    slug = "models--" + model_id.replace("/", "--")
    return (cache / slug).exists()


def preflight(model_id: str, out_dir: Path) -> str:
    device = detect_device()
    log(f"device={device}")
    if device == "cpu":
        fail(
            "no_gpu",
            "No CUDA or Apple Metal GPU. Image jobs need a GPU (NVIDIA Docker or native MPS).",
        )

    target = out_dir if out_dir.exists() else out_dir.parent
    free_gb = shutil.disk_usage(target).free / (1024 ** 3)
    need = MIN_DISK_CACHED_GB if model_cached(model_id) else MIN_DISK_FIRST_GB
    log(f"disk_free={free_gb:.1f}GB need>={need}GB cached={model_cached(model_id)}")
    if free_gb < need:
        fail(
            "no_disk",
            f"Need about {need} GB free for weights/output (have {free_gb:.1f} GB).",
        )

    if device == "cuda":
        free_mb = nvidia_free_vram_mb()
        if free_mb is not None:
            log(f"vram_free={free_mb}MB")
            if free_mb < MIN_VRAM_MB:
                fail(
                    "no_vram",
                    f"Need ~{MIN_VRAM_MB} MB free VRAM for Klein 4B (have {free_mb} MB).",
                )
            if free_mb < 8000:
                log("warning: under 8 GB free VRAM — small model / offload")
    elif device == "mps":
        ram = unified_ram_gb()
        if ram is not None:
            log(f"unified_ram={ram:.1f}GB")
            if ram < MIN_UNIFIED_GB:
                fail(
                    "no_vram",
                    f"Apple Silicon needs at least {MIN_UNIFIED_GB} GB unified memory (have {ram:.0f} GB).",
                )
            if ram < 16:
                log("warning: 8–16 GB unified — generation may be slow or offload to RAM")
    return device


def _to_device(pipe, device: str, tight: bool) -> None:
    if device != "cuda":
        pipe.to(device)
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        fn = getattr(pipe, name, None)
        if callable(fn):
            try:
                fn()
                log(f"{name} on")
            except Exception:
                pass
    if tight:
        fn = getattr(pipe, "enable_sequential_cpu_offload", None)
        if callable(fn):
            try:
                fn()
                log("cuda sequential_cpu_offload enabled")
                return
            except Exception:
                pass
    fn = getattr(pipe, "enable_model_cpu_offload", None)
    if callable(fn):
        try:
            fn()
            log("cuda cpu_offload enabled")
            return
        except Exception:
            pass
    pipe.to("cuda")


def load_pipeline(model_id: str, family: str, device: str, img2img: bool):
    import torch
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    tight = False
    if device == "cuda":
        total = nvidia_total_vram_mb()
        tight = total is not None and total < KLEIN_MIN_TOTAL_MB
    log(f"loading {model_id} family={family} dtype={dtype} (first use of this model downloads weights)")
    try:
        if family == "sdxl":
            if img2img:
                from diffusers import AutoPipelineForImage2Image
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model_id, torch_dtype=dtype
                )
            else:
                from diffusers import AutoPipelineForText2Image
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id, torch_dtype=dtype
                )
        else:
            from diffusers import Flux2KleinPipeline
            pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as e:
        fail("model_load", f"Could not load {model_id}: {e}")
    _to_device(pipe, device, tight=tight or family == "klein")
    return pipe


def _accepted_call_kwargs(pipe, raw: dict) -> dict:
    """Klein is not SDXL: no `strength`. Only pass names the pipeline accepts."""
    import inspect
    try:
        params = set(inspect.signature(pipe.__call__).parameters)
    except (TypeError, ValueError):
        params = set()
    if not params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(pipe.__call__).parameters.values()):
        # Still drop known-invalid SDXL leftovers.
        raw.pop("strength", None)
        return raw
    return {k: v for k, v in raw.items() if k in params and k != "self"}


def run_pipe(pipe, job: dict, init_image, family: str):
    prompt = job["prompt"]
    steps = int(job.get("steps") or 4)
    default_g = 0.0 if family == "sdxl" else 1.0
    guidance = float(job.get("guidance") if job.get("guidance") is not None else default_g)
    if family == "sdxl" and guidance == 1.0:
        guidance = 0.0  # Turbo is trained for CFG 0
    width = int(job.get("width") or 1024)
    height = int(job.get("height") or 1024)
    width = max(64, (width // 16) * 16)
    height = max(64, (height // 16) * 16)
    seed = int(job.get("seed") or -1)
    n = int(job.get("num_images") or 1)
    negative = (job.get("negative_prompt") or "") or None

    import torch
    gen = None
    if seed >= 0:
        try:
            gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        except Exception:
            gen = torch.Generator().manual_seed(seed)

    def on_step(*args, **kwargs):
        # Diffusers versions pass (pipe, step, t, cb) or (step, t, cb).
        step_idx = 0
        for a in args:
            if isinstance(a, int):
                step_idx = a
                break
        total = max(1, steps)
        frac = min(0.99, (int(step_idx) + 1) / total)
        print(f"[image] progress {frac:.3f} step={int(step_idx)+1}/{total}", flush=True)
        if args and isinstance(args[-1], dict):
            return args[-1]
        return kwargs.get("callback_kwargs") or {}

    if init_image is not None and init_image.size != (width, height):
        init_image = init_image.resize((width, height))
        log(f"resized init to {width}x{height}")

    raw = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "num_images_per_prompt": n,
        "generator": gen,
        "callback_on_step_end": on_step,
    }
    if negative:
        raw["negative_prompt"] = negative
    if init_image is not None:
        raw["image"] = init_image
        if family == "sdxl":
            raw["strength"] = float(job.get("strength") or 0.75)
            log(f"img2img sdxl strength={raw['strength']} size={width}x{height} steps={steps}")
        else:
            log(f"img2img klein reference-edit size={width}x{height} steps={steps}")
    else:
        log(f"txt2img size={width}x{height} steps={steps} family={family}")

    kwargs = _accepted_call_kwargs(pipe, raw)
    log(f"pipeline kwargs={sorted(kwargs)}")
    try:
        return pipe(**kwargs)
    except TypeError as e:
        log(f"retry without extras ({e})")
        for k in ("negative_prompt", "callback_on_step_end", "num_images_per_prompt"):
            kwargs.pop(k, None)
        return pipe(**kwargs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True, help="Path to job JSON")
    args = p.parse_args()
    job_path = Path(args.job)
    job = json.loads(job_path.read_text())

    out_dir = Path(job.get("output_dir") or (job_path.parent / "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = job.get("model_id") or job.get("model") or DEFAULT_MODEL
    mode = (job.get("mode") or "txt2img").lower()

    device = preflight(requested, out_dir)
    model_id, family = pick_model_id(str(requested), device)
    log(f"model={model_id} family={family} mode={mode}")
    pipe = load_pipeline(model_id, family, device, img2img=(mode == "img2img"))

    init = None
    if mode == "img2img":
        src = job.get("init_image")
        if not src or not Path(src).is_file():
            fail("scene_error", "img2img requires init_image file")
        from PIL import Image
        init = Image.open(src).convert("RGB")
        log(f"init_image={src} {init.size[0]}x{init.size[1]}")

    try:
        result = run_pipe(pipe, job, init, family)
    except SystemExit:
        raise
    except Exception as e:
        msg = str(e).lower()
        oom = "out of memory" in msg or "oom" in msg
        if oom and family == "klein":
            log("Klein OOM — falling back to SDXL Turbo")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            del pipe
            model_id, family = SDXL_TURBO_ID, "sdxl"
            pipe = load_pipeline(model_id, family, device, img2img=(mode == "img2img"))
            result = run_pipe(pipe, job, init, family)
        elif oom:
            fail("no_vram", str(e))
        else:
            fail("image_failed", str(e))

    images = getattr(result, "images", None) or []
    if not images:
        fail("image_failed", "Pipeline returned no images")
    written: list[str] = []
    for i, im in enumerate(images):
        dest = out_dir / f"image_{i+1:02d}.png"
        im.save(dest, "PNG")
        written.append(str(dest))
        log(f"wrote {dest.name} ({dest.stat().st_size} bytes)")
    print(f"[image] progress 1.000 files={len(written)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"[image] ERROR [image_failed] {e}", flush=True)
        raise SystemExit(1)
