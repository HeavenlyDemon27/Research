"""
Training loop and checkpoint handling for Mini-BART E1.
E1 uses supervised cross-entropy only.

global_step counts successful optimizer updates (not microbatches).
Checkpoints are written only at optimizer-update boundaries so no
partially accumulated gradients need to be serialized.
"""
import os
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 2
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 8
CHECKPOINT_EVERY_STEPS = 500
LATEST_CHECKPOINT_NAME = "latest_checkpoint.pt"
BEST_CHECKPOINT_NAME = "best_checkpoint.pt"
CHECKPOINT_FORMAT_VERSION = 2


def create_optimizer_and_scheduler(model, total_training_steps):
    no_decay = ["bias", "layer_norm", "layernorm"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(x in n.lower() for x in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(x in n.lower() for x in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=LEARNING_RATE)
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps)
    return optimizer, scheduler


def train_one_epoch(
    model, dataloader, optimizer, scheduler, device, scaler=None,
    global_step=0, train_sampler=None, output_dir=None, epoch=0,
    best_val_loss=float("inf"), epoch_loss_sum=0.0, epoch_examples=0,
    history=None, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    model.train()
    total_loss = float(epoch_loss_sum)
    total_examples = int(epoch_examples)
    use_amp = device.type == "cuda"
    total_batches = len(dataloader)
    if total_batches == 0:
        raise ValueError("Training dataloader is empty.")

    optimizer.zero_grad(set_to_none=True)
    group_size = 0
    microbatches_in_group = 0

    progress_bar = tqdm(enumerate(dataloader, start=1), total=total_batches, desc="Training")
    for batch_position, batch in progress_bar:
        # Start a new accumulation group. The final group may be smaller.
        if microbatches_in_group == 0:
            group_size = min(gradient_accumulation_steps, total_batches - batch_position + 1)
            optimizer.zero_grad(set_to_none=True)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            outputs = model(**batch)
            raw_loss = outputs.loss
            scaled_loss = raw_loss / group_size

        if scaler is not None and use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        batch_size = batch["input_ids"].size(0)
        total_loss += raw_loss.detach().item() * batch_size
        total_examples += batch_size
        microbatches_in_group += 1

        did_update = False
        if microbatches_in_group == group_size:
            if scaler is not None and use_amp:
                old_scale = scaler.get_scale()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                did_update = scaler.get_scale() >= old_scale
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                did_update = True

            optimizer.zero_grad(set_to_none=True)
            microbatches_in_group = 0

            if did_update:
                scheduler.step()
                global_step += 1

                if (global_step % CHECKPOINT_EVERY_STEPS == 0
                    and train_sampler is not None and output_dir is not None
                    and train_sampler.position < len(train_sampler.data_source)):
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch,
                        batch_position, global_step, best_val_loss,
                        train_sampler, output_dir,
                        epoch_loss_sum=total_loss,
                        epoch_examples=total_examples, history=history,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                    )

        progress_bar.set_postfix(
            loss=f"{raw_loss.detach().item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
            updates=global_step,
        )

    if microbatches_in_group != 0:
        raise RuntimeError("Training ended with unstepped accumulated gradients.")
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
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                loss = model(**batch).loss
            n = batch["input_ids"].size(0)
            total_loss += loss.item() * n
            total_examples += n
    if total_examples == 0:
        raise ValueError("Validation dataloader is empty.")
    return total_loss / total_examples


def save_checkpoint(
    model, optimizer, scheduler, scaler, epoch, batch_position,
    global_step, best_val_loss, train_sampler, output_dir,
    filename=LATEST_CHECKPOINT_NAME, epoch_loss_sum=0.0,
    epoch_examples=0, history=None,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch, "batch_position": batch_position,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "epoch_loss_sum": float(epoch_loss_sum),
        "epoch_examples": int(epoch_examples),
        "history": list(history or []),
        "sampler_state_dict": train_sampler.state_dict(),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "config": {
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS, "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
            "global_step_definition": "successful_optimizer_updates",
            "objective": "supervised_cross_entropy_only",
        },
    }
    path = os.path.join(output_dir, filename)
    tmp = path + ".tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)
    print(f"Checkpoint saved | epoch={epoch + 1} | optimizer_updates={global_step} | microbatch_position={batch_position}")


def load_checkpoint(
    checkpoint_path, model, optimizer, scheduler, scaler,
    train_sampler, device,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version", 0) < CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Legacy checkpoint uses microbatch-based steps. Start E1 fresh; do not resume it with the accumulation-aware trainer.")
    saved_accum = checkpoint.get("gradient_accumulation_steps")
    if saved_accum != gradient_accumulation_steps:
        raise ValueError(f"Checkpoint accumulation={saved_accum}, requested={gradient_accumulation_steps}. Resume with the original setting.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    train_sampler.load_state_dict(checkpoint["sampler_state_dict"])
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return {
        "epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
        "best_val_loss": checkpoint["best_val_loss"],
        "epoch_loss_sum": checkpoint.get("epoch_loss_sum", 0.0),
        "epoch_examples": checkpoint.get("epoch_examples", 0),
        "history": checkpoint.get("history", []),
    }


def train_model(
    model, train_dataloader, val_dataloader, device, output_dir,
    num_epochs=NUM_EPOCHS, resume_from=None,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)
    train_sampler = train_dataloader.sampler
    if not hasattr(train_sampler, "state_dict"):
        raise TypeError("Training DataLoader must use a sampler with state_dict() for checkpoint resumption.")

    updates_per_epoch = (len(train_dataloader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_updates = updates_per_epoch * num_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, total_updates)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    global_step = start_epoch = 0
    epoch_loss_sum = 0.0
    epoch_examples = 0
    history = []

    if resume_from is not None:
        state = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler,
            train_sampler, device, gradient_accumulation_steps,
        )
        start_epoch = state["epoch"]
        global_step = state["global_step"]
        best_val_loss = state["best_val_loss"]
        epoch_loss_sum = state["epoch_loss_sum"]
        epoch_examples = state["epoch_examples"]
        history = state["history"]
        print(f"Resuming epoch {start_epoch + 1}, optimizer update {global_step}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, global_step = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, device,
            scaler=scaler, global_step=global_step, train_sampler=train_sampler,
            output_dir=output_dir, epoch=epoch, best_val_loss=best_val_loss,
            epoch_loss_sum=epoch_loss_sum, epoch_examples=epoch_examples,
            history=history, gradient_accumulation_steps=gradient_accumulation_steps,
        )
        epoch_loss_sum = 0.0
        epoch_examples = 0
        val_loss = validate_one_epoch(model, val_dataloader, device)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1, 0,
                global_step, best_val_loss, train_sampler, output_dir,
                filename=BEST_CHECKPOINT_NAME, history=history,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            print("New best validation checkpoint saved.")

        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1, 0,
            global_step, best_val_loss, train_sampler, output_dir,
            filename=LATEST_CHECKPOINT_NAME, history=history,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    return {
        "history": history, "best_val_loss": best_val_loss,
        "best_checkpoint": os.path.join(output_dir, BEST_CHECKPOINT_NAME),
        "latest_checkpoint": os.path.join(output_dir, LATEST_CHECKPOINT_NAME),
        "global_step": global_step,
    }
