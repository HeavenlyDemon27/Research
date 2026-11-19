
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import shutil
import tempfile
import sys
from pathlib import Path

# Add the Mini-BART project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from types import SimpleNamespace

from src.data import StatefulRandomSampler
from src.train import train_model, load_checkpoint
import src.train as train_module


# 1. Tiny synthetic dataset
class DummyDataset(Dataset):
    def __init__(self, size=16):
        self.data = [
            {
                "input_ids": torch.randint(1, 20, (8,)),
                "attention_mask": torch.ones(8, dtype=torch.long),
                "labels": torch.randint(1, 20, (8,)),
            }
            for _ in range(size)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 2. Tiny model that returns a loss
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 16)
        self.classifier = nn.Linear(16, 20)

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.embedding(input_ids)
        logits = self.classifier(hidden)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, 20),
            labels.reshape(-1),
        )

        return SimpleNamespace(loss=loss)


def main():
    torch.manual_seed(42)

    dataset = DummyDataset(size=16)

    sampler = StatefulRandomSampler(
        dataset,
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        generator=torch.Generator().manual_seed(123),
    )

    val_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    model = DummyModel()
    device = torch.device("cpu")

    # Force frequent periodic checkpoints for this small test.
    train_module.CHECKPOINT_EVERY_STEPS = 2

    with tempfile.TemporaryDirectory() as output_dir:

        print("\n========== DUMMY TRAINING ==========")

        results = train_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            output_dir=output_dir,
            num_epochs=2,
        )

        print("\n========== RESULTS ==========")
        print("Global step:", results["global_step"])
        print("Best validation loss:", results["best_val_loss"])

        # 3. Check checkpoint files
        latest_path = os.path.join(
            output_dir, "latest_checkpoint.pt"
        )
        best_path = os.path.join(
            output_dir, "best_checkpoint.pt"
        )

        assert os.path.exists(latest_path), "Latest checkpoint missing"
        assert os.path.exists(best_path), "Best checkpoint missing"

        print("\nPASS: Both checkpoint files exist.")

        # 4. Verify checkpoint contents
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )

        required_keys = [
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "sampler_state_dict",
            "rng_state",
            "epoch",
            "global_step",
        ]

        for key in required_keys:
            assert key in checkpoint, f"Missing checkpoint key: {key}"

        print("PASS: Checkpoint contains all required states.")

        # 5. Test loading into a fresh model and optimizer
        restored_model = DummyModel()

        restored_sampler = StatefulRandomSampler(
            dataset,
            generator=torch.Generator().manual_seed(42),
        )

        restored_loader = DataLoader(
            dataset,
            batch_size=4,
            sampler=restored_sampler,
            generator=torch.Generator().manual_seed(123),
        )

        restored_optimizer, restored_scheduler = (
            train_module.create_optimizer_and_scheduler(
                restored_model,
                len(restored_loader) * 2,
            )
        )

        restored_scaler = torch.cuda.amp.GradScaler(enabled=False)

        restored_state = load_checkpoint(
            checkpoint_path=latest_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            scaler=restored_scaler,
            train_sampler=restored_sampler,
            device=device,
        )

        assert restored_state["epoch"] == 2
        assert restored_state["global_step"] == 8

        print("PASS: Checkpoint successfully loaded.")
        print("PASS: Epoch and global step restored.")

        print("\n========== ALL DUMMY TESTS PASSED ==========")


if __name__ == "__main__":
    main()
