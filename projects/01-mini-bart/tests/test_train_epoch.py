import torch
import torch.nn as nn
from types import SimpleNamespace
from src.train import (
    create_optimizer_and_scheduler,
    train_one_epoch,
)


class TinyDummyModel(nn.Module):
    def __init__(self, vocab_size=20, hidden_size=16):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, labels):
        # Embed the input tokens
        embeddings = self.embedding(input_ids)

        # Mask padding and calculate mean pooled representation
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Convert pooled representation to vocabulary logits
        logits = self.output_layer(pooled)

        # Repeat logits across target sequence length
        logits = logits.unsqueeze(1).expand(
            -1, labels.size(1), -1
        )

        # Calculate cross-entropy, ignoring padding labels (-100)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return SimpleNamespace(loss=loss)
def main():
    torch.manual_seed(42)

    model = TinyDummyModel()

    # Three synthetic batches
    dataloader = [
        {
            "input_ids": torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "labels": torch.tensor([[7, 8, 9], [4, 5, -100]]),
        },
        {
            "input_ids": torch.tensor([[3, 4, 5, 0], [6, 7, 8, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
            "labels": torch.tensor([[8, 9, 10], [5, 6, 7]]),
        },
        {
            "input_ids": torch.tensor([[4, 5, 6, 0], [7, 8, 9, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
            "labels": torch.tensor([[9, 10, 11], [6, 7, 8]]),
        },
    ]

    device = torch.device("cpu")

    # Save original parameters for update verification
    original_parameters = [
        p.detach().clone() for p in model.parameters()
    ]

    total_training_steps = len(dataloader)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        total_training_steps,
    )

    print("--- Training Epoch Test ---")
    print("Device:", device)
    print("Batches:", len(dataloader))
    print("Parameters before training saved.")

    avg_loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # Check parameter updates
    parameters_updated = any(
        not torch.equal(old, new.detach())
        for old, new in zip(original_parameters, model.parameters())
    )

    # Verify optimizer updates occurred for each batch
    optimizer_steps = optimizer.state[next(model.parameters())]["step"].item()

    print("\n--- Verification Results ---")
    print("Average training loss:", avg_loss)
    print("Loss is finite:", torch.isfinite(torch.tensor(avg_loss)).item())
    print("Parameters updated:", parameters_updated)
    print("Optimizer steps:", optimizer_steps)
    print("Scheduler steps:", scheduler.last_epoch)

    assert torch.isfinite(torch.tensor(avg_loss))
    assert parameters_updated
    assert optimizer_steps == len(dataloader)
    assert scheduler.last_epoch == len(dataloader)

    print("\ntrain_one_epoch verification passed!")


if __name__ == "__main__":
    main()    