
"""
Training loop and checkpoint handling for Mini-BART E1.
E1 uses supervised cross-entropy only.
"""

import os
import torch

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm


# Training configuration
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0


def create_optimizer_and_scheduler(model, total_training_steps):

    no_decay = ["bias", "layer_norm", "layernorm"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n.lower() for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
    )

    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    return optimizer, scheduler


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    scaler=None,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    use_amp = device.type == "cuda"

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:

        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=use_amp,
            dtype=torch.float16,
        ):
            outputs = model(**batch)
            loss = outputs.loss

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

        scheduler.step()

        batch_size = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    if total_examples == 0:
        raise ValueError("Training dataloader is empty.")

    return total_loss / total_examples


def validate_one_epoch(model, dataloader, device):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    use_amp = device.type == "cuda"

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Validation"):

            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            with torch.cuda.amp.autocast(
                enabled=use_amp,
                dtype=torch.float16,
            ):
                outputs = model(**batch)
                loss = outputs.loss

            batch_size = batch["input_ids"].size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Validation dataloader is empty.")

    return total_loss / total_examples


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "objective": "supervised_cross_entropy_only",
        },
    }

    path = os.path.join(output_dir, "best_checkpoint.pt")

    temporary_path = path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)

    print(f"Best checkpoint saved: {path}")


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    device,
    output_dir,
    num_epochs=NUM_EPOCHS,
):
    model.to(device)

    total_steps = len(train_dataloader) * num_epochs

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        total_steps,
    )

    use_amp = device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    history = []

    for epoch in range(num_epochs):

        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 50}")

        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            scaler,
        )

        val_loss = validate_one_epoch(
            model,
            val_dataloader,
            device,
        )

        print(f"Training loss:   {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_val_loss,
                output_dir,
            )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_checkpoint": os.path.join(
            output_dir,
            "best_checkpoint.pt",
        ),
    }
