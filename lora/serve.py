"""Keep one 4-bit model loaded and answer evaluation prompts from /session/inbox.

The agent uses this on a friend's GPU. A MilliCloud host does not. The process
exits after EVAL_IDLE_SECONDS with no new prompt, or when /session/unload appears.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import infer

ROOT = Path(os.environ.get("EVAL_SESSION_ROOT", "/session"))
IDLE = int(os.environ.get("EVAL_IDLE_SECONDS", "600"))
INBOX = ROOT / "inbox"
OUT = ROOT / "out"


def log(msg: str) -> None:
    print(f"[lora] {msg}", flush=True)


def write_meta(payload: dict) -> None:
    path = ROOT / "meta.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def meta(state: str, job: dict, loaded_at: float, last_used: float) -> None:
    now = time.time()
    write_meta({
        "state": state,
        "model_id": job.get("model_id") or "",
        "revision": job.get("model_revision") or "main",
        "adapter": job.get("adapter_label") or "adapter",
        "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(loaded_at)),
        "idle_seconds": max(0, int(now - last_used)),
        "hold_seconds": IDLE,
        "host": "friend",
    })


def answer(model, tokenizer, prompt: str, with_adapter: bool) -> str:
    if with_adapter:
        return infer.generate(model, tokenizer, prompt)
    with model.disable_adapter():
        return infer.generate(model, tokenizer, prompt)


def main() -> None:
    job = json.loads((ROOT / "session.json").read_text(encoding="utf-8"))
    model_id = str(job.get("model_id") or "")
    revision = str(job.get("model_revision") or "main")
    adapter_dir = Path(job.get("adapter_dir") or "/session/adapter")
    if not model_id:
        raise SystemExit("session.json is missing model_id")
    INBOX.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"cache loading {model_id}@{revision} adapter={job.get('adapter_label') or adapter_dir}")
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = infer.load_base_model(model_id, revision, bnb)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    model.config.use_cache = True
    loaded_at = time.time()
    last_used = loaded_at
    log(f"cache loaded {model_id} hold={IDLE}s")
    meta("loaded", job, loaded_at, last_used)

    while True:
        if (ROOT / "unload").exists():
            log("cache unload requested")
            meta("unloading", job, loaded_at, last_used)
            return
        pending = sorted(INBOX.glob("*.json"))
        if not pending:
            idle = time.time() - last_used
            meta("loaded", job, loaded_at, last_used)
            if idle >= IDLE:
                log(f"cache idle {int(idle)}s, unloading")
                meta("expired", job, loaded_at, last_used)
                return
            time.sleep(1)
            continue
        path = pending[0]
        try_id = path.stem
        try:
            req = json.loads(path.read_text(encoding="utf-8"))
            prompt = str(req.get("prompt") or "").strip()
            if not prompt:
                raise RuntimeError("prompt is empty")
            log(f"cache answer {try_id}")
            base_text = answer(model, tokenizer, prompt, with_adapter=False)
            adapter_text = answer(model, tokenizer, prompt, with_adapter=True)
            (OUT / f"{try_id}.json").write_text(
                json.dumps({"base": base_text, "adapter": adapter_text}, ensure_ascii=False),
                encoding="utf-8",
            )
            log(f"cache wrote {try_id}")
        except Exception as exc:
            (OUT / f"{try_id}.json").write_text(
                json.dumps({"error": str(exc)[:1900]}),
                encoding="utf-8",
            )
            log(f"cache error {try_id}: {exc}")
        finally:
            path.unlink(missing_ok=True)
            last_used = time.time()
            meta("loaded", job, loaded_at, last_used)


if __name__ == "__main__":
    main()
