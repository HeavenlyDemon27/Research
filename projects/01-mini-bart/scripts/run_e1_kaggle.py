"""
Kaggle entry point for Mini-BART E1 supervised training.

Run from the project root:
python scripts/run_e1_kaggle.py \
  --tokenized-dir /kaggle/working/mini_bart_tokenized_512_128_v1 \
  --output-dir /kaggle/working/mini_bart_E1
"""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, BartForConditionalGeneration

from src.data import create_dataloaders
from src.model import prepare_e1_student
from src.train import (
    train_model,
    NUM_EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    MAX_GRAD_NORM,
    CHECKPOINT_FORMAT_VERSION,
)


EXPECTED_TRAIN = 287_113
EXPECTED_VALIDATION = 13_368
TEACHER_ID = "facebook/bart-large-cnn"
BATCH_SIZE = 1
NUM_WORKERS = 0
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-id", default=TEACHER_ID)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_existing_checkpoint(path, train_size):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "format_version",
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "sampler_state_dict",
        "rng_state",
        "config",
        "gradient_accumulation_steps",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(
            f"Existing latest checkpoint is incomplete: {sorted(missing)}. "
            "Preserve it; do not overwrite it."
        )

    if checkpoint["format_version"] < CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            "Existing checkpoint format is incompatible with the current trainer."
        )

    if checkpoint["gradient_accumulation_steps"] != GRADIENT_ACCUMULATION_STEPS:
        raise RuntimeError("Checkpoint gradient accumulation does not match E1.")

    config = checkpoint["config"]
    expected_config = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "objective": "supervised_cross_entropy_only",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"Checkpoint config mismatch for {key}: "
                f"found {config.get(key)!r}, expected {expected!r}."
            )

    sampler_state = checkpoint["sampler_state_dict"]
    permutation = sampler_state.get("permutation")
    position = int(sampler_state.get("position", -1))

    if permutation is not None and len(permutation) != train_size:
        raise RuntimeError(
            "Checkpoint sampler belongs to a different training-set size."
        )
    if position < 0 or (
        permutation is not None and position > len(permutation)
    ):
        raise RuntimeError("Checkpoint sampler position is invalid.")

    return checkpoint


def main():
    args = parse_args()
    tokenized_dir = Path(args.tokenized_dir)
    output_dir = Path(args.output_dir)

    if not tokenized_dir.is_dir():
        raise FileNotFoundError(f"Tokenized dataset not found: {tokenized_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. E1 must run on a GPU.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )

    seed_everything(SEED)

    print("\nLoading saved tokenized train/validation data...")
    tokenized_dataset = load_from_disk(str(tokenized_dir))

    if "train" not in tokenized_dataset or "validation" not in tokenized_dataset:
        raise KeyError("Saved tokenized dataset must contain train and validation.")

    if "test" in tokenized_dataset and len(tokenized_dataset["test"]) != 0:
        raise RuntimeError(
            "A non-empty test split was found. E1 must not access test examples."
        )

    if len(tokenized_dataset["train"]) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN:,} train examples, "
            f"found {len(tokenized_dataset['train']):,}."
        )
    if len(tokenized_dataset["validation"]) != EXPECTED_VALIDATION:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION:,} validation examples, "
            f"found {len(tokenized_dataset['validation']):,}."
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_id, use_fast=True)

    print("Creating fresh DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        tokenized_dataset,
        tokenizer,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    if train_loader.sampler.position != 0 or train_loader.sampler.permutation is not None:
        raise RuntimeError("Training sampler was consumed before E1 launch.")
    if len(train_loader) != EXPECTED_TRAIN:
        raise RuntimeError("Unexpected train DataLoader length.")
    if len(val_loader) != EXPECTED_VALIDATION:
        raise RuntimeError("Unexpected validation DataLoader length.")
    if len(test_loader) != 0:
        raise RuntimeError("Test DataLoader must be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_checkpoint.pt"
    best_path = output_dir / "best_checkpoint.pt"

    if best_path.exists() and not latest_path.exists():
        raise RuntimeError(
            "best_checkpoint.pt exists but latest_checkpoint.pt is missing. "
            "Inspect the output directory; refusing to start over."
        )

    resume_from = None
    if latest_path.exists():
        checkpoint = validate_existing_checkpoint(
            latest_path,
            len(train_loader.dataset),
        )
        resume_from = str(latest_path)
        print(
            "Compatible E1 checkpoint found:",
            f"epoch={checkpoint['epoch'] + 1}",
            f"optimizer_updates={checkpoint['global_step']}",
            f"sampler_position={checkpoint['sampler_state_dict']['position']}",
        )
        del checkpoint
    else:
        print("No existing checkpoint. E1 will start from teacher initialization.")

    print("\n========== E1 CONFIGURATION ==========")
    print("Train examples:", f"{len(train_loader.dataset):,}")
    print("Validation examples:", f"{len(val_loader.dataset):,}")
    print("Test examples:", len(test_loader.dataset))
    print("Microbatch size:", BATCH_SIZE)
    print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)
    print("Effective batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Warmup ratio:", WARMUP_RATIO)
    print("Objective: supervised cross-entropy only")
    print("Teacher:", args.teacher_id)
    print("Output:", output_dir)
    print("Resume:", resume_from or "Fresh run")
    print("======================================")

    print("\nLoading teacher on CPU...")
    teacher = BartForConditionalGeneration.from_pretrained(
        args.teacher_id,
        low_cpu_mem_usage=True,
    )
    teacher.eval()

    if teacher.config.encoder_layers != 12 or teacher.config.decoder_layers != 12:
        raise RuntimeError("Expected the 12-encoder/12-decoder BART-large teacher.")

    print("Building Mini-BART and transferring teacher layers [0, 2, 4, 6, 8, 10]...")
    student = prepare_e1_student(teacher)

    if student.config.encoder_layers != 6 or student.config.decoder_layers != 6:
        raise RuntimeError("Student architecture is not 6 encoder + 6 decoder layers.")
    if not all(parameter.requires_grad for parameter in student.parameters()):
        raise RuntimeError("At least one student parameter is unexpectedly frozen.")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    print("Student parameters:", f"{sum(p.numel() for p in student.parameters()):,}")
    print("Starting E1 training...")
    result = train_model(
        model=student,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        output_dir=str(output_dir),
        num_epochs=NUM_EPOCHS,
        resume_from=resume_from,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    summary = {
        "experiment": "E1",
        "objective": "supervised_cross_entropy_only",
        "teacher": args.teacher_id,
        "student_encoder_layers": 6,
        "student_decoder_layers": 6,
        "train_examples": EXPECTED_TRAIN,
        "validation_examples": EXPECTED_VALIDATION,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "seed": SEED,
        "history": result["history"],
        "best_val_loss": result["best_val_loss"],
        "global_step": result["global_step"],
        "best_checkpoint": result["best_checkpoint"],
        "latest_checkpoint": result["latest_checkpoint"],
    }

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== E1 TRAINING FINISHED ==========")
    print(json.dumps(summary, indent=2))
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()
