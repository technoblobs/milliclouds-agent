"""Compare the base model with a finished LoRA adapter. One prompt, two answers.

The agent copies this file into the job workdir and runs it with the trainer image.
Stdout is for the worker log. The answers are written to try_result.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_base_model(model_id: str, revision: str, bnb):
    """4-bit load. Gemma 4 is registered as a multimodal class, not a plain causal LM."""
    kwargs = dict(
        revision=revision,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def log(msg: str) -> None:
    print(f"[lora] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[lora] ERROR [try_failed] {msg}", flush=True)
    sys.exit(1)


def generate(model, tokenizer, prompt: str) -> str:
    text = f"### Prompt:\n{prompt.strip()}\n\n### Response:\n"
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1500)
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.inference_mode():
        out = model.generate(
            **ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][ids["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    model_id = str(job.get("model_id") or "")
    if not model_id:
        fail("job is missing model_id")
    revision = str(job.get("model_revision") or "main")
    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        fail("job is missing a prompt")
    adapter_dir = Path(job.get("adapter_dir") or "/work/adapter")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        fail("adapter_model.safetensors is missing")

    log(f"compare model={model_id}@{revision}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        model = load_base_model(model_id, revision, bnb)
    except Exception as e:
        fail(str(e))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True

    log("generating base")
    base = generate(model, tokenizer, prompt)
    log("generating adapter")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    adapter = generate(model, tokenizer, prompt)

    out = Path(job.get("output") or "/work/out/try_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"base": base, "adapter": adapter}, ensure_ascii=False),
        encoding="utf-8",
    )
    log("wrote compare")


if __name__ == "__main__":
    main()
