
"""
Training loop and checkpoint handling for Mini-BART E1.
E1 uses supervised cross-entropy only.
"""

import os
import torch
import random
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm


# Training configuration
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 2
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
# Checkpoint configuration
CHECKPOINT_EVERY_STEPS = 500
LATEST_CHECKPOINT_NAME = "latest_checkpoint.pt"
BEST_CHECKPOINT_NAME = "best_checkpoint.pt"

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
    global_step=0,
    train_sampler=None,
    output_dir=None,
    epoch=0,
    best_val_loss=float("inf"),
    epoch_loss_sum=0.0,
    epoch_examples=0,
):
    model.train()

    # Carry these accumulators across mid-epoch checkpoint/resume.
    total_loss = float(epoch_loss_sum)
    total_examples = int(epoch_examples)

    use_amp = device.type == "cuda"

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_position, batch in enumerate(progress_bar, start=1):

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
            old_scale = scaler.get_scale()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

            # GradScaler reduces its scale when an optimizer
            # update is skipped because of FP16 overflow.
            if scaler.get_scale() >= old_scale:
                scheduler.step()

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
        global_step += 1

        if (
            global_step % CHECKPOINT_EVERY_STEPS == 0
            and train_sampler is not None
            and output_dir is not None
            and train_sampler.position < len(train_sampler.data_source)
        ):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                batch_position=batch_position,
                global_step=global_step,
                best_val_loss=best_val_loss,
                train_sampler=train_sampler,
                output_dir=output_dir,
                epoch_loss_sum=total_loss,
                epoch_examples=total_examples,
            )

    if total_examples == 0:
        raise ValueError("Training dataloader is empty.")

    return total_loss / total_examples, global_step


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
    batch_position,
    global_step,
    best_val_loss,
    train_sampler,
    output_dir,
    filename=LATEST_CHECKPOINT_NAME,
    epoch_loss_sum=0.0,
    epoch_examples=0,
):
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        # Progress
        "epoch": epoch,
        "batch_position": batch_position,
        "global_step": global_step,

        # Model and training state
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),

        # Experiment state
        "best_val_loss": best_val_loss,
        "epoch_loss_sum": float(epoch_loss_sum),
        "epoch_examples": int(epoch_examples),

        # Sampler state (reproducible data order)
        "sampler_state_dict": train_sampler.state_dict(),

        # Random number generator states
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },

        # Hyperparameters
        "config": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
            "objective": "supervised_cross_entropy_only",
        },
    }

    path = os.path.join(
        output_dir,
        filename,
    )

    temporary_path = path + ".tmp"

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)
    display_epoch = epoch if batch_position == 0 else epoch + 1

    print(
        f"Checkpoint saved | "
        f"epoch={epoch + 1} | "
        f"step={global_step} | "
        f"batch_position={batch_position}"
    )
def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    scaler,
    train_sampler,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Restore model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore training components
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if scaler is not None:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    # Restore data sampling position/order
    train_sampler.load_state_dict(
        checkpoint["sampler_state_dict"]
    )

    # Restore random number generator states
    rng_state = checkpoint["rng_state"]

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])

    if (
        torch.cuda.is_available()
        and rng_state["cuda"] is not None
    ):
        torch.cuda.set_rng_state_all(
            rng_state["cuda"]
        )

    print(
        f"Checkpoint restored | "
        f"epoch={checkpoint['epoch'] + 1} | "
        f"step={checkpoint['global_step']} | "
        f"batch_position={checkpoint['batch_position']}"
    )

    return {
        "epoch": checkpoint["epoch"],
        "batch_position": checkpoint["batch_position"],
        "global_step": checkpoint["global_step"],
        "best_val_loss": checkpoint["best_val_loss"],
        "epoch_loss_sum": checkpoint.get("epoch_loss_sum", 0.0),
        "epoch_examples": checkpoint.get("epoch_examples", 0),
    }    
def train_model(
    model,
    train_dataloader,
    val_dataloader,
    device,
    output_dir,
    num_epochs=NUM_EPOCHS,
    resume_from=None,
):
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)

    # Retrieve the stateful sampler used by the training DataLoader.
    train_sampler = train_dataloader.sampler

    if not hasattr(train_sampler, "state_dict"):
        raise TypeError(
            "Training DataLoader must use a sampler with state_dict() "
            "for checkpoint resumption."
        )

    total_steps = len(train_dataloader) * num_epochs

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        total_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0
    epoch_loss_sum = 0.0
    epoch_examples = 0
    history = []

    # Resume from an existing checkpoint if requested.
    if resume_from is not None:
        print(f"Resuming training from: {resume_from}")

        resume_state = load_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_sampler=train_sampler,
            device=device,
        )

        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        best_val_loss = resume_state["best_val_loss"]
        epoch_loss_sum = resume_state["epoch_loss_sum"]
        epoch_examples = resume_state["epoch_examples"]

        print(
            f"Resuming at epoch {start_epoch + 1}, "
            f"global step {global_step}"
        )

    # Training loop.
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            global_step=global_step,
            train_sampler=train_sampler,
            output_dir=output_dir,
            epoch=epoch,
            best_val_loss=best_val_loss,
            epoch_loss_sum=epoch_loss_sum,
            epoch_examples=epoch_examples,
        )

        # The epoch is now complete; its accumulators are no longer needed
        # for a subsequent epoch-boundary checkpoint.
        epoch_loss_sum = 0.0
        epoch_examples = 0

        val_loss = validate_one_epoch(
            model,
            val_dataloader,
            device,
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # Save the best checkpoint when validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                batch_position=0,
                global_step=global_step,
                best_val_loss=best_val_loss,
                train_sampler=train_sampler,
                output_dir=output_dir,
                filename=BEST_CHECKPOINT_NAME,
                epoch_loss_sum=0.0,
                epoch_examples=0,
            )

            print("New best validation checkpoint saved.")

        # Always save the latest state at the epoch boundary.
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            batch_position=0,
            global_step=global_step,
            best_val_loss=best_val_loss,
            train_sampler=train_sampler,
            output_dir=output_dir,
            filename=LATEST_CHECKPOINT_NAME,
            epoch_loss_sum=0.0,
            epoch_examples=0,
        )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_checkpoint": os.path.join(
            output_dir,
            BEST_CHECKPOINT_NAME,
        ),
        "latest_checkpoint": os.path.join(
            output_dir,
            LATEST_CHECKPOINT_NAME,
        ),
        "global_step": global_step,
    }
