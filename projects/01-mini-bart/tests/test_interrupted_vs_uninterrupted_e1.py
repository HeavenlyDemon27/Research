import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import src.train as train_module
from src.data import StatefulRandomSampler


class ToyDataset(Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(900)
        self.rows = [
            {
                "input_ids": torch.randint(0, 10, (5,), generator=g),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "labels": torch.randint(0, 10, (5,), generator=g),
            }
            for _ in range(16)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 8)
        self.proj = nn.Linear(8, 10)

    def forward(self, input_ids, attention_mask, labels):
        logits = self.proj(self.emb(input_ids))
        loss = nn.functional.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1))
        return SimpleNamespace(loss=loss)


def make_loader(dataset):
    sampler = StatefulRandomSampler(dataset, generator=torch.Generator().manual_seed(42))
    return DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0,
                      generator=torch.Generator().manual_seed(123))


def tensor_state_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def run_full(dataset, out_dir):
    torch.manual_seed(1234)
    model = ToyModel()
    loader = make_loader(dataset)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    result = train_module.train_model(
        model, loader, val_loader, torch.device("cpu"), out_dir,
        num_epochs=2, gradient_accumulation_steps=2,
    )
    return model, result


def main():
    dataset = ToyDataset()
    train_module.CHECKPOINT_EVERY_STEPS = 2

    with tempfile.TemporaryDirectory() as tmp:
        full_dir = Path(tmp) / "full"
        interrupted_dir = Path(tmp) / "interrupted"
        full_model, full_result = run_full(dataset, full_dir)

        torch.manual_seed(1234)
        interrupted_model = ToyModel()
        interrupted_loader = make_loader(dataset)
        validation_loader = DataLoader(dataset, batch_size=2, shuffle=False)

        original_train_one_epoch = train_module.train_one_epoch

        def interrupt_after_four_microbatches(*args, **kwargs):
            original_loader = kwargs["dataloader"]
            class StopAfterFour:
                def __len__(self):
                    return len(original_loader)
                def __iter__(self):
                    it = iter(original_loader)
                    for _ in range(4):
                        yield next(it)
                    raise RuntimeError("SIMULATED_RUNTIME_INTERRUPTION")
            kwargs["dataloader"] = StopAfterFour()
            return original_train_one_epoch(*args, **kwargs)

        train_module.train_one_epoch = interrupt_after_four_microbatches
        try:
            try:
                train_module.train_model(
                    interrupted_model, interrupted_loader, validation_loader,
                    torch.device("cpu"), interrupted_dir, num_epochs=2,
                    gradient_accumulation_steps=2,
                )
                raise AssertionError("Expected simulated interruption.")
            except RuntimeError as exc:
                assert "SIMULATED_RUNTIME_INTERRUPTION" in str(exc)
        finally:
            train_module.train_one_epoch = original_train_one_epoch

        latest = interrupted_dir / "latest_checkpoint.pt"
        assert latest.exists(), "Checkpoint was not written before interruption."
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        assert checkpoint["global_step"] == 2
        assert checkpoint["gradient_accumulation_steps"] == 2
        assert checkpoint["sampler_state_dict"]["position"] == 8

        resumed_model = ToyModel()
        resumed_loader = make_loader(dataset)
        resumed_result = train_module.train_model(
            resumed_model, resumed_loader, validation_loader,
            torch.device("cpu"), interrupted_dir, num_epochs=2,
            resume_from=str(latest), gradient_accumulation_steps=2,
        )

        assert tensor_state_equal(full_model.state_dict(), resumed_model.state_dict())
        assert full_result["global_step"] == resumed_result["global_step"]
        assert full_result["history"] == resumed_result["history"]
        print("PASS: accumulation-aware interrupted/resumed training is identical.")
        print("Optimizer updates:", full_result["global_step"])
        print("Final model parameters and history match exactly.")


if __name__ == "__main__":
    main()
