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
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[lora] {msg}", flush=True)


def _human_bytes(n: float) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def hub_download_bar():
    """tqdm subclass that logs Hub downloads as normal lines.

    huggingface_hub disables its bar when stderr is not a terminal, which is
    how the trainer runs under the agent. A line at the start, every 10%, and
    every 20 seconds while a large file is moving is what the console and the
    website can show.
    """
    cached = getattr(hub_download_bar, "_cls", None)
    if cached is not None:
        return cached

    from tqdm.auto import tqdm

    class HubDownloadBar(tqdm):
        saw_large = False
        _lock = threading.Lock()
        _MIN_BYTES = 8 * 1024 * 1024

        def __init__(self, *args, **kwargs):
            self._announced = False
            self._last_pct = -1
            self._last_log_at = 0.0
            kwargs.pop("name", None)
            kwargs["disable"] = False
            kwargs["file"] = io.StringIO()
            super().__init__(*args, **kwargs)
            self._report()

        def update(self, n=1):
            shown = super().update(n)
            self._report()
            return shown

        def refresh(self, *args, **kwargs):
            shown = super().refresh(*args, **kwargs)
            self._report()
            return shown

        def close(self):
            self._report(final=True)
            return super().close()

        def _label(self) -> str:
            desc = str(self.desc or "model file")
            if "incomplete" in desc.lower():
                return "model files"
            return desc

        def _report(self, final: bool = False) -> None:
            total = float(self.total or 0)
            if total < self._MIN_BYTES:
                return
            done = float(self.n or 0)
            pct = min(100, int(100 * done / total))
            now = time.monotonic()
            with self._lock:
                if not self._announced:
                    self._announced = True
                    type(self).saw_large = True
                    self._last_pct = 0
                    self._last_log_at = now
                    log(
                        f"downloading {self._label()} ({_human_bytes(total)}). "
                        "A first run can sit here for several minutes."
                    )
                if self._last_pct >= 100:
                    return
                bucket = pct // 10 * 10
                heartbeat = now - self._last_log_at >= 20 and pct < 100
                if not final and bucket < self._last_pct + 10 and not heartbeat:
                    return
                if pct == 100 or final:
                    self._last_pct = 100
                elif bucket > self._last_pct:
                    self._last_pct = bucket
                self._last_log_at = now
            note = " still downloading" if heartbeat and pct < 100 else ""
            log(
                f"download {self._label()} {pct}% "
                f"({_human_bytes(done)} / {_human_bytes(total)}){note}"
            )

    hub_download_bar._cls = HubDownloadBar
    return HubDownloadBar


def fail(code: str, msg: str) -> None:
    print(f"[lora] ERROR [{code}] {msg}", flush=True)
    raise SystemExit(2)


def import_failure(exc: BaseException) -> str:
    """Transformers 5 hides the missing module behind a generic AutoModel error."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{type(cur).__name__}: {cur}"
        if text not in parts:
            parts.append(text)
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts)


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


def _pair_from_obj(obj: dict, prompt_field: str, response_field: str) -> tuple[str, str]:
    """Accept prompt/response, Alpaca instruction/output, or chat messages."""
    prompt = str(obj.get(prompt_field) or "").strip()
    response = str(obj.get(response_field) or "").strip()
    if prompt and response:
        return prompt, response
    instruction = str(obj.get("instruction") or "").strip()
    output = str(obj.get("output") or "").strip()
    extra = str(obj.get("input") or "").strip()
    if instruction and output:
        if extra:
            instruction = f"{instruction}\n\n{extra}"
        return instruction, output
    messages = obj.get("messages")
    if isinstance(messages, list):
        user = ""
        assistant = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict)
                )
            content = str(content or "").strip()
            if role in ("user", "human") and content:
                user = content
            elif role in ("assistant", "gpt") and content:
                assistant = content
        if user and assistant:
            return user, assistant
    return "", ""


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
            prompt, response = _pair_from_obj(obj, prompt_field, response_field)
            if not prompt or not response:
                fail(
                    "scene_error",
                    f"line {lineno} needs prompt/response, instruction/output, or messages",
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


def load_base_model(model_id: str, revision: str, bnb):
    """4-bit load. Gemma 4 is registered as a multimodal class, not a plain causal LM."""
    import torch
    from transformers import AutoModelForCausalLM

    bar = hub_download_bar()
    bar.saw_large = False
    kwargs = dict(
        revision=revision,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        tqdm_class=bar,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


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
    # The image defaults to root. Keep adapter files readable by the host user.
    os.umask(0o022)
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
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as e:
        fail(
            "no_libs",
            "training libraries missing ("
            + import_failure(e)
            + "). Rebuild millicloud-lora:qwen and wait for 'trainer torch ok'.",
        )

    prompt_field = str(dataset_spec.get("prompt_field") or "prompt")
    response_field = str(dataset_spec.get("response_field") or "response")
    rows = load_rows(
        Path(job["dataset"]),
        prompt_field,
        response_field,
        limit=int(dataset_spec.get("max_rows") or 50_000),
        min_rows=int(dataset_spec.get("min_rows") or 2),
    )
    eval_tail = int(job.get("eval_tail") or 0)
    if eval_tail < 1 or eval_tail >= len(rows):
        eval_tail = max(1, len(rows) // 20)  # last 5% when the user did not pass a validation file
    train_rows, eval_rows = rows[:-eval_tail], rows[-eval_tail:]
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
    loss_on = str(job.get("loss_on") or "full").lower()
    if loss_on not in ("full", "response"):
        loss_on = "full"
    early_stop = bool(job.get("early_stop", True))
    patience = max(1, min(8, int(job.get("early_stop_patience") or 2)))
    out_dir = Path(job["output_dir"])
    ckpt_dir = Path(job.get("checkpoint_dir") or (out_dir.parent / "ckpt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"model={model_id}@{model_revision} rank={rank} lr={lr} "
        f"epochs={epochs} max_seq={max_seq} loss_on={loss_on} "
        f"early_stop={early_stop} modules={','.join(modules)}"
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    log(
        f"fetching {model_id}@{model_revision}. "
        "A download prints its percent below. A cached model stays quiet while the weights load onto the GPU."
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, use_fast=True)
        model = load_base_model(model_id, model_revision, bnb)
    except Exception as e:
        fail("model_load", str(e))
    if not hub_download_bar().saw_large:
        log(f"model weights already cached for {model_id}")
    log("weights loaded, preparing the adapter")

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

    def encode_row(row: dict) -> dict:
        prompt_text = f"### Prompt:\n{row['prompt']}\n\n### Response:\n"
        eos = tokenizer.eos_token or ""
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        response_ids = tokenizer(row["response"] + eos, add_special_tokens=False).input_ids
        ids = (prompt_ids + response_ids)[:max_seq]
        if loss_on == "response":
            cut = min(len(prompt_ids), len(ids))
            labels = [-100] * cut + ids[cut:]
        else:
            labels = list(ids)
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
        }

    train_ds = Dataset.from_list([encode_row(row) for row in train_rows])
    eval_ds = Dataset.from_list([encode_row(row) for row in eval_rows])

    class Progress(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            total = max(1, state.max_steps or 1)
            frac = min(0.99, state.global_step / total)
            print(f"[lora] progress {frac:.4f} step={state.global_step}/{total}", flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            if logs.get("loss") is not None:
                parts = [f"step={state.global_step}", f"loss={float(logs['loss']):.6f}"]
                if logs.get("grad_norm") is not None:
                    parts.append(f"grad_norm={float(logs['grad_norm']):.4f}")
                if logs.get("learning_rate") is not None:
                    parts.append(f"learning_rate={float(logs['learning_rate']):.8f}")
                print("[lora] metric " + " ".join(parts), flush=True)
            if logs.get("eval_loss") is not None:
                print(
                    f"[lora] eval_loss step={state.global_step} value={float(logs['eval_loss']):.6f}",
                    flush=True,
                )

        def on_save(self, args, state, control, **kwargs):
            print(f"[lora] checkpoint {args.output_dir}", flush=True)

    updates_per_epoch = max(1, len(train_rows) // max(1, batch_size * grad_accum))
    planned_steps = max(1, updates_per_epoch * epochs)
    eval_steps = save_steps if planned_steps > save_steps * 2 else max(1, planned_steps // 4)
    callbacks = [Progress()]
    eval_strategy = "no"
    if early_stop:
        eval_strategy = "steps"
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=0.0))

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        # Transformers 5 takes the warmup fraction here. warmup_ratio was removed.
        warmup_steps=warmup,
        logging_steps=10,
        save_strategy="steps",
        save_steps=eval_steps if early_stop else save_steps,
        save_total_limit=3,
        bf16=True,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=early_stop,
        metric_for_best_model="eval_loss" if early_stop else None,
        greater_is_better=False if early_stop else None,
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
        eval_dataset=eval_ds if early_stop else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding=True),
        callbacks=callbacks,
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
    taken = int(trainer.state.global_step or 0)
    planned = int(trainer.state.max_steps or taken)
    early = early_stop and planned > 0 and taken < planned
    if early:
        log(f"early_stop step={taken} kept_best=1")
    model.save_pretrained(out_dir)
    result = {
        "val_loss": val_loss,
        "global_step": taken,
        "early_stopped": early,
        "best_step": int(getattr(trainer.state, "best_global_step", None) or taken),
        "loss_on": loss_on,
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
