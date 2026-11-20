
"""
Knowledge distillation training loop for Mini-BART E2.

The teacher is frozen.
Only the student is optimized.

Objective:
    L_total = L_CE + lambda_kd * L_KD
"""

import os
import torch
from tqdm.auto import tqdm

from src.train import create_optimizer_and_scheduler, MAX_GRAD_NORM
from src.distillation import combined_distillation_loss


# --------------------------------------------------
# E2 Configuration
# --------------------------------------------------

KD_TEMPERATURE = 2.0
KD_LAMBDA = 0.5

KD_NUM_EPOCHS = 2

KD_CHECKPOINT_DIR = "experiments/E2/checkpoints"


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_kd(
    teacher,
    student,
    dataloader,
    device,
    temperature=KD_TEMPERATURE,
    lambda_kd=KD_LAMBDA,
):
    """
    Evaluate teacher/student combined loss on validation data.
    No gradients are computed.
    """

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


# --------------------------------------------------
# Train one epoch
# --------------------------------------------------

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
):
    """
    Train the student for one epoch using knowledge distillation.
    """

    teacher.eval()
    student.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_examples = 0
    if len(dataloader) == 0:
        raise ValueError("Training dataloader is empty.")

    use_amp = device.type == "cuda"

    progress_bar = tqdm(dataloader, desc="E2 Training")

    for batch in progress_bar:

        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        # ------------------------------------------
        # Teacher forward: frozen, no gradients
        # ------------------------------------------

        with torch.no_grad():

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                teacher_outputs = teacher(**batch)

        # ------------------------------------------
        # Student forward: gradients enabled
        # ------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            student_outputs = student(**batch)

        # ------------------------------------------
        # Compute combined CE + KD objective
        # ------------------------------------------

        losses = combined_distillation_loss(
            teacher_logits=teacher_outputs.logits,
            student_logits=student_outputs.logits,
            labels=batch["labels"],
            temperature=temperature,
            lambda_kd=lambda_kd,
        )

        loss = losses["loss"]

        # ------------------------------------------
        # Backpropagation: student only
        # ------------------------------------------

        if use_amp:

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

        scheduler.step()

        # ------------------------------------------
        # Logging
        # ------------------------------------------

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

    if total_examples == 0:
        raise ValueError("Training dataloader is empty.")

    return {
        "loss": total_loss / total_examples,
        "ce_loss": total_ce_loss / total_examples,
        "kd_loss": total_kd_loss / total_examples,
        "global_step": global_step,
    }


# --------------------------------------------------
# Checkpoint
# --------------------------------------------------

def save_kd_checkpoint(
    student,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_val_loss,
    output_dir,
    filename,
    temperature,
    lambda_kd,
):
    """
    Save E2 student and training state.
    """

    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "temperature": temperature,
            "lambda_kd": lambda_kd,
            "num_epochs": KD_NUM_EPOCHS,
            "objective": "cross_entropy_plus_knowledge_distillation",
            "teacher": "facebook/bart-large-cnn",
            "student_encoder_layers": 6,
            "student_decoder_layers": 6,
        },
    }

    path = os.path.join(output_dir, filename)
    temp_path = path + ".tmp"

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)

    print(f"E2 checkpoint saved: {path}")


# --------------------------------------------------
# Main E2 training function
# --------------------------------------------------

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
):
    """
    Full E2 training workflow.

    Teacher: frozen
    Student: trainable
    """

    teacher.to(device)
    student.to(device)

    teacher.eval()

    # Enforce teacher freezing.
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    # Enforce student trainability.
    for parameter in student.parameters():
        parameter.requires_grad = True

    os.makedirs(output_dir, exist_ok=True)

    # Optimizer receives student parameters ONLY.
    optimizer, scheduler = create_optimizer_and_scheduler(
        student,
        len(train_dataloader) * num_epochs,
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
    )

    best_val_loss = float("inf")
    global_step = 0
    history = []

    for epoch in range(num_epochs):

        print(f"\n========== E2 Epoch {epoch + 1}/{num_epochs} ==========")

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

        # Save best student based on validation total loss.
        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]

            save_kd_checkpoint(
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                best_val_loss=best_val_loss,
                output_dir=output_dir,
                filename="best_kd_checkpoint.pt",
                temperature=temperature,
                lambda_kd=lambda_kd,
            )

        # Save latest student at every epoch boundary.
        save_kd_checkpoint(
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            best_val_loss=best_val_loss,
            output_dir=output_dir,
            filename="latest_kd_checkpoint.pt",
            temperature=temperature,
            lambda_kd=lambda_kd,
        )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "best_checkpoint": os.path.join(
            output_dir,
            "best_kd_checkpoint.pt",
        ),
        "latest_checkpoint": os.path.join(
            output_dir,
            "latest_kd_checkpoint.pt",
        ),
    }
