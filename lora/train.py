#!/usr/bin/env python3
"""MilliCloud LoRA trainer — QLoRA on one NVIDIA GPU.

Reads a job JSON, trains one adapter on the full dataset, writes the adapter
and a train_result.json. Prints lines the agent already knows how to ship:

    [lora] progress 0.2500 step=100/400
    [lora] checkpoint /work/ckpt
    [lora] val_loss 1.234567

A resumed job passes ``resume`` pointing at a previous Trainer checkpoint.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"[lora] {msg}", flush=True)


def fail(code: str, msg: str) -> None:
    print(f"[lora] ERROR [{code}] {msg}", flush=True)
    raise SystemExit(2)


def _free_vram_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return int(float(out.stdout.strip().splitlines()[0].split(",")[0]))
    except ValueError:
        return None


def load_rows(
    path: Path,
    prompt_field: str,
    response_field: str,
    limit: int,
    min_rows: int,
) -> list[dict]:
    if not path.is_file():
        fail("scene_error", f"dataset missing: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                fail("scene_error", f"line {lineno} is not JSON: {e}")
            if not isinstance(obj, dict):
                fail("scene_error", f"line {lineno} must be an object")
            prompt = str(obj.get(prompt_field) or "").strip()
            response = str(obj.get(response_field) or "").strip()
            if not prompt or not response:
                fail(
                    "scene_error",
                    f"line {lineno} needs non-empty {prompt_field} and {response_field}",
                )
            rows.append({"prompt": prompt, "response": response})
            if len(rows) >= limit:
                log(f"dataset capped at {limit} rows")
                break
    if len(rows) < min_rows:
        fail("scene_error", f"dataset needs at least {min_rows} non-empty rows")
    return rows


def as_text(row: dict) -> str:
    return f"### Prompt:\n{row['prompt']}\n\n### Response:\n{row['response']}"


def find_resume(path: str | None) -> str | None:
    if not path:
        return None
    root = Path(path)
    if not root.exists():
        log(f"resume path missing, training from step 0 ({root})")
        return None
    if (root / "trainer_state.json").is_file():
        return str(root)
    found = sorted(p for p in root.glob("checkpoint-*") if p.is_dir())
    if not found:
        log("resume archive had no checkpoint, training from step 0")
        return None
    log(f"resuming from {found[-1].name}")
    return str(found[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = Path(args.job)
    try:
        job = json.loads(job_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        fail("scene_error", f"bad job file: {e}")

    hardware = job.get("hardware") or {}
    dataset_spec = job.get("dataset_spec") or {}
    recipe = job.get("recipe") or {}
    min_disk_gb = int(hardware.get("min_disk_gb") or 30)
    min_vram_mb = int(hardware.get("min_vram_mb") or 12000)
    free = shutil.disk_usage(job_path.parent).free
    if free < min_disk_gb * 1024**3:
        fail(
            "no_disk",
            f"this template needs {min_disk_gb} GB free for weights and checkpoints, "
            f"have {free // 1024**3} GB",
        )

    try:
        import torch
    except ImportError:
        fail("no_libs", "PyTorch is not installed. Build millicloud-lora:qwen (see lora.md).")
    if not torch.cuda.is_available():
        fail("no_gpu", "LoRA training requires an NVIDIA GPU with CUDA")

    vram = _free_vram_mb()
    if vram is not None and vram < min_vram_mb:
        fail(
            "no_vram",
            f"this template needs about {min_vram_mb} MB free VRAM, this GPU has {vram} MB",
        )

    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as e:
        fail("no_libs", f"training libraries missing ({e}). Build millicloud-lora:qwen.")

    prompt_field = str(dataset_spec.get("prompt_field") or "prompt")
    response_field = str(dataset_spec.get("response_field") or "response")
    rows = load_rows(
        Path(job["dataset"]),
        prompt_field,
        response_field,
        limit=int(dataset_spec.get("max_rows") or 50_000),
        min_rows=int(dataset_spec.get("min_rows") or 2),
    )
    hold = max(1, len(rows) // 20)  # last 5%
    train_rows, eval_rows = rows[:-hold], rows[-hold:]
    log(f"rows train={len(train_rows)} eval={len(eval_rows)}")

    trial = job.get("trial") or {}
    rank = int(trial.get("rank") or job.get("rank") or 16)
    lr = float(trial.get("lr") or job.get("lr") or 2e-4)
    epochs = int(trial.get("epochs") or job.get("epochs") or 1)
    max_seq = int(trial.get("max_seq") or job.get("max_seq") or 2048)
    model_id = str(job.get("model_id") or "")
    if not model_id:
        fail("scene_error", "job is missing model_id from its template")
    model_revision = str(job.get("model_revision") or "main")
    modules = recipe.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
    dropout = float(recipe.get("dropout") if recipe.get("dropout") is not None else 0.05)
    alpha_scale = int(recipe.get("alpha_scale") or 2)
    batch_size = int(recipe.get("batch_size") or 1)
    grad_accum = int(recipe.get("grad_accum") or 8)
    warmup = float(recipe.get("warmup_ratio") if recipe.get("warmup_ratio") is not None else 0.03)
    save_steps = int(recipe.get("save_steps") or 200)
    out_dir = Path(job["output_dir"])
    ckpt_dir = Path(job.get("checkpoint_dir") or (out_dir.parent / "ckpt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"model={model_id}@{model_revision} rank={rank} lr={lr} "
        f"epochs={epochs} max_seq={max_seq} modules={','.join(modules)}"
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        fail("model_load", str(e))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=rank * alpha_scale,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(modules),
        ),
    )

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            [as_text({"prompt": p, "response": r}) for p, r in zip(batch["prompt"], batch["response"])],
            truncation=True,
            max_length=max_seq,
        )

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["prompt", "response"])
    eval_ds = Dataset.from_list(eval_rows).map(tokenize, batched=True, remove_columns=["prompt", "response"])

    class Progress(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            total = max(1, state.max_steps or 1)
            frac = min(0.99, state.global_step / total)
            print(f"[lora] progress {frac:.4f} step={state.global_step}/{total}", flush=True)

        def on_save(self, args, state, control, **kwargs):
            print(f"[lora] checkpoint {args.output_dir}", flush=True)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        bf16=True,
        eval_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[Progress()],
    )
    resume = find_resume(job.get("resume"))
    try:
        trainer.train(resume_from_checkpoint=resume)
    except RuntimeError as e:
        text = str(e).lower()
        if "out of memory" in text or "cuda" in text:
            fail("no_vram", str(e))
        fail("lora_failed", str(e))

    eval_metrics = trainer.evaluate(eval_dataset=eval_ds)
    val_loss = float(eval_metrics.get("eval_loss"))
    log(f"val_loss {val_loss:.6f}")
    model.save_pretrained(out_dir)
    result = {
        "val_loss": val_loss,
        "global_step": int(trainer.state.global_step or 0),
        "model_id": model_id,
        "rank": rank,
        "lr": lr,
        "epochs": epochs,
        "max_seq": max_seq,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    log(f"progress 1.0000 step={result['global_step']}/{result['global_step'] or 1}")
    log(f"wrote adapter under {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[lora] ERROR [lora_failed] {e}", flush=True)
        sys.exit(2)
