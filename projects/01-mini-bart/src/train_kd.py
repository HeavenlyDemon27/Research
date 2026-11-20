
"""
Knowledge distillation training loop for Mini-BART E2.

Supports:
- Frozen teacher and trainable student
- Combined CE + KD objective
- AMP training
- Mid-epoch checkpoints
- Reproducible training resume
"""

import os
import random
import numpy as np
import torch

from tqdm.auto import tqdm

from src.train import create_optimizer_and_scheduler, MAX_GRAD_NORM
from src.distillation import combined_distillation_loss


# ============================================================
# 1. CONFIGURATION
# ============================================================

KD_TEMPERATURE = 2.0
KD_LAMBDA = 0.5
KD_NUM_EPOCHS = 2

KD_CHECKPOINT_DIR = "experiments/E2/checkpoints"

KD_CHECKPOINT_EVERY_STEPS = 500

LATEST_CHECKPOINT_NAME = "latest_kd_checkpoint.pt"
BEST_CHECKPOINT_NAME = "best_kd_checkpoint.pt"


def get_kd_config(num_epochs, temperature, lambda_kd):
    """Return the experiment configuration saved in checkpoints."""

    return {
        "temperature": temperature,
        "lambda_kd": lambda_kd,
        "num_epochs": num_epochs,
        "objective": "cross_entropy_plus_knowledge_distillation",
        "teacher": "facebook/bart-large-cnn",
        "student_encoder_layers": 6,
        "student_decoder_layers": 6,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "max_grad_norm": MAX_GRAD_NORM,
    }


# ============================================================
# 2. RANDOM NUMBER GENERATOR STATE
# ============================================================

def capture_rng_state():
    """Capture RNG states required for reproducible resumption."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(rng_state):
    """Restore all saved RNG states."""

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])

    if (
        torch.cuda.is_available()
        and rng_state["cuda"] is not None
    ):
        torch.cuda.set_rng_state_all(rng_state["cuda"])


# ============================================================
# 3. ATOMIC CHECKPOINT SAVING
# ============================================================

def atomic_torch_save(checkpoint, path):
    """Write a checkpoint atomically."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    temporary_path = path + ".tmp"

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def save_kd_checkpoint(
    student,
    optimizer,
    scheduler,
    scaler,
    train_sampler,
    epoch,
    global_step,
    best_val_loss,
    output_dir,
    filename,
    config,
    include_training_state=True,
):
    """
    Save either a full recovery checkpoint or a best-model checkpoint.

    epoch is the NEXT epoch to execute for a recovery checkpoint.
    For a mid-epoch checkpoint, this is the currently running epoch.
    """

    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        "student_state_dict": student.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    if include_training_state:
        if train_sampler is None:
            raise ValueError(
                "A sampler is required for a recovery checkpoint."
            )

        checkpoint.update({
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "sampler_state_dict": train_sampler.state_dict(),
            "rng_state": capture_rng_state(),
        })

    path = os.path.join(output_dir, filename)

    atomic_torch_save(checkpoint, path)

    print(
        f"Checkpoint saved | {filename} | "
        f"next_epoch={epoch} | step={global_step}"
    )

    return path


# ============================================================
# 4. CHECKPOINT LOADING
# ============================================================

def load_kd_checkpoint(
    checkpoint_path,
    student,
    optimizer,
    scheduler,
    scaler,
    train_sampler,
    device,
    expected_config,
):
    """Restore a complete E2 training state."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_config = checkpoint["config"]

    if saved_config != expected_config:
        raise ValueError(
            "Checkpoint configuration does not match the current "
            f"training configuration.\nSaved: {saved_config}\n"
            f"Current: {expected_config}"
        )

    required_keys = [
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "sampler_state_dict",
        "rng_state",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(
                f"Checkpoint is missing required recovery state: {key}"
            )

    student.load_state_dict(
        checkpoint["student_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    train_sampler.load_state_dict(
        checkpoint["sampler_state_dict"]
    )

    # Restore RNG last, after model and optimizer setup.
    # This must happen before creating the next DataLoader iterator.
    restore_rng_state(checkpoint["rng_state"])

    print(
        f"Resumed E2 | next_epoch={checkpoint['epoch']} | "
        f"global_step={checkpoint['global_step']}"
    )

    return {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "best_val_loss": checkpoint["best_val_loss"],
    }


# ============================================================
# 5. VALIDATION
# ============================================================

def validate_kd(
    teacher,
    student,
    dataloader,
    device,
    temperature=KD_TEMPERATURE,
    lambda_kd=KD_LAMBDA,
):
    teacher.eval()
    student.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_examples = 0

    use_amp = device.type == "cuda"

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="E2 Validation"):

            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                teacher_outputs = teacher(**batch)
                student_outputs = student(**batch)

            losses = combined_distillation_loss(
                teacher_logits=teacher_outputs.logits,
                student_logits=student_outputs.logits,
                labels=batch["labels"],
                temperature=temperature,
                lambda_kd=lambda_kd,
            )

            batch_size = batch["input_ids"].size(0)

            total_loss += losses["loss"].item() * batch_size
            total_ce_loss += losses["ce_loss"].item() * batch_size
            total_kd_loss += losses["kd_loss"].item() * batch_size

            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Validation dataloader is empty.")

    return {
        "loss": total_loss / total_examples,
        "ce_loss": total_ce_loss / total_examples,
        "kd_loss": total_kd_loss / total_examples,
    }


# ============================================================
# 6. ONE TRAINING EPOCH
# ============================================================

def train_kd_one_epoch(
    teacher,
    student,
    dataloader,
    optimizer,
    scheduler,
    device,
    scaler,
    global_step=0,
    temperature=KD_TEMPERATURE,
    lambda_kd=KD_LAMBDA,
    checkpoint_callback=None,
    checkpoint_every_steps=KD_CHECKPOINT_EVERY_STEPS,
):
    teacher.eval()
    student.train()

    if len(dataloader) == 0:
        raise ValueError("Training dataloader is empty.")

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_examples = 0

    use_amp = device.type == "cuda"

    progress_bar = tqdm(dataloader, desc="E2 Training")

    for batch in progress_bar:

        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        # Teacher forward pass.
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                teacher_outputs = teacher(**batch)

        # Student forward pass.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            student_outputs = student(**batch)

        # CE + KD objective.
        losses = combined_distillation_loss(
            teacher_logits=teacher_outputs.logits,
            student_logits=student_outputs.logits,
            labels=batch["labels"],
            temperature=temperature,
            lambda_kd=lambda_kd,
        )

        loss = losses["loss"]

        # Backpropagation.
        if use_amp:

            old_scale = scaler.get_scale()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

            # A reduced scale means the optimizer update was skipped.
            update_succeeded = scaler.get_scale() >= old_scale

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()
            update_succeeded = True

        # Scheduler advances only after a successful optimizer update.
        if update_succeeded:
            scheduler.step()

        batch_size = batch["input_ids"].size(0)

        total_loss += losses["loss"].item() * batch_size
        total_ce_loss += losses["ce_loss"].item() * batch_size
        total_kd_loss += losses["kd_loss"].item() * batch_size
        total_examples += batch_size

        global_step += 1

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{losses['ce_loss'].item():.4f}",
            kd=f"{losses['kd_loss'].item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        # Save only at consumed-batch boundaries.
        # Avoid saving after the final sample of an epoch because
        # the sampler may be transitioning to its next epoch.
        if (
            checkpoint_callback is not None
            and checkpoint_every_steps > 0
            and global_step % checkpoint_every_steps == 0
        ):
            checkpoint_callback(global_step)

    if total_examples == 0:
        raise ValueError("Training dataloader is empty.")

    return {
        "loss": total_loss / total_examples,
        "ce_loss": total_ce_loss / total_examples,
        "kd_loss": total_kd_loss / total_examples,
        "global_step": global_step,
    }


# ============================================================
# 7. COMPLETE E2 TRAINING WORKFLOW
# ============================================================

def train_kd_model(
    teacher,
    student,
    train_dataloader,
    val_dataloader,
    device,
    output_dir=KD_CHECKPOINT_DIR,
    num_epochs=KD_NUM_EPOCHS,
    temperature=KD_TEMPERATURE,
    lambda_kd=KD_LAMBDA,
    resume_from=None,
    checkpoint_every_steps=KD_CHECKPOINT_EVERY_STEPS,
):
    """
    Train E2 with full checkpoint recovery support.

    resume_from:
        Path to latest_kd_checkpoint.pt, or None for a fresh run.
    """

    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps cannot be negative.")

    if getattr(train_dataloader, "num_workers", 0) != 0:
        raise ValueError(
            "Exact E2 resume currently requires num_workers=0 "
            "to avoid DataLoader prefetch position discrepancies."
        )

    teacher.to(device)
    student.to(device)

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    for parameter in student.parameters():
        parameter.requires_grad = True

    train_sampler = train_dataloader.sampler

    if not hasattr(train_sampler, "state_dict"):
        raise TypeError(
            "E2 training requires a stateful training sampler."
        )

    os.makedirs(output_dir, exist_ok=True)

    total_steps = len(train_dataloader) * num_epochs

    optimizer, scheduler = create_optimizer_and_scheduler(
        student,
        total_steps,
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
    )

    config = get_kd_config(
        num_epochs=num_epochs,
        temperature=temperature,
        lambda_kd=lambda_kd,
    )

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0
    history = []

    # Resume if requested.
    if resume_from is not None:

        resume_state = load_kd_checkpoint(
            checkpoint_path=resume_from,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_sampler=train_sampler,
            device=device,
            expected_config=config,
        )

        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        best_val_loss = resume_state["best_val_loss"]

    # Training epochs.
    for epoch in range(start_epoch, num_epochs):

        print(
            f"\n========== E2 Epoch {epoch + 1}/{num_epochs} =========="
        )

        # Mid-epoch checkpoint callback.

        def save_latest_mid_epoch(step):

            # Do not create a mid-epoch checkpoint when the sampler
            # has already reached the final example of this epoch.
            if (
                train_sampler.permutation is None
                or train_sampler.position >= len(train_sampler.permutation)
            ):
                return

            save_kd_checkpoint(
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_sampler=train_sampler,
                epoch=epoch,
                global_step=step,
                best_val_loss=best_val_loss,
                output_dir=output_dir,
                filename=LATEST_CHECKPOINT_NAME,
                config=config,
                include_training_state=True,
            )


        train_metrics = train_kd_one_epoch(
            teacher=teacher,
            student=student,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            global_step=global_step,
            temperature=temperature,
            lambda_kd=lambda_kd,
            checkpoint_callback=save_latest_mid_epoch,
            checkpoint_every_steps=checkpoint_every_steps,
        )

        global_step = train_metrics["global_step"]

        val_metrics = validate_kd(
            teacher=teacher,
            student=student,
            dataloader=val_dataloader,
            device=device,
            temperature=temperature,
            lambda_kd=lambda_kd,
        )

        print(
            f"\nEpoch {epoch + 1} Results\n"
            f"Train Total: {train_metrics['loss']:.4f} | "
            f"CE: {train_metrics['ce_loss']:.4f} | "
            f"KD: {train_metrics['kd_loss']:.4f}\n"
            f"Val Total: {val_metrics['loss']:.4f} | "
            f"CE: {val_metrics['ce_loss']:.4f} | "
            f"KD: {val_metrics['kd_loss']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": val_metrics,
        })

        # Save best model separately.
        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]

            save_kd_checkpoint(
                student=student,
                optimizer=None,
                scheduler=None,
                scaler=None,
                train_sampler=None,
                epoch=epoch + 1,
                global_step=global_step,
                best_val_loss=best_val_loss,
                output_dir=output_dir,
                filename=BEST_CHECKPOINT_NAME,
                config=config,
                include_training_state=False,
            )

        # Save complete latest recovery checkpoint at epoch boundary.
        save_kd_checkpoint(
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_sampler=train_sampler,
            epoch=epoch + 1,
            global_step=global_step,
            best_val_loss=best_val_loss,
            output_dir=output_dir,
            filename=LATEST_CHECKPOINT_NAME,
            config=config,
            include_training_state=True,
        )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "best_checkpoint": os.path.join(
            output_dir,
            BEST_CHECKPOINT_NAME,
        ),
        "latest_checkpoint": os.path.join(
            output_dir,
            LATEST_CHECKPOINT_NAME,
        ),
    }
