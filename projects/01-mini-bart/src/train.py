"""
Training loop and checkpoint handling for Mini-BART E1.
"""

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
    """
    Create AdamW optimizer and linear learning-rate scheduler.
    """

    # Parameters that should NOT receive weight decay
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight"]

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
):
    """
    Train the model for one complete epoch.
    """

    model.train()

    total_loss = 0.0
    total_examples = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:

        # Move batch tensors to the selected device
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # Reset gradients from previous batch
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(**batch)

        loss = outputs.loss

        # Backpropagation
        loss.backward()

        # Clip gradients to prevent excessively large updates
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        # Update model parameters
        optimizer.step()

        # Update learning rate
        scheduler.step()

        # Accumulate loss
        batch_size = batch["input_ids"].size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    average_loss = total_loss / total_examples

    return average_loss