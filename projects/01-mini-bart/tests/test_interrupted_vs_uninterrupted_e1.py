import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import copy
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
        }

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
    return DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0,
                     generator=torch.Generator().manual_seed(123))


def tensor_state_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def run_full(dataset, out_dir):
    torch.manual_seed(1234)
    model = ToyModel()
    loader = make_loader(dataset)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return model, train_module.train_model(
        model, loader, val_loader, torch.device("cpu"), out_dir, num_epochs=2
    )


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
        validation_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        original_train_one_epoch = train_module.train_one_epoch
        original_save_checkpoint = train_module.save_checkpoint
        original_validate = train_module.validate_one_epoch

        # Simulate runtime interruption immediately after step 2, after checkpoint is written.
        def interrupt_after_two_batches(*args, **kwargs):
            original_loader = kwargs["dataloader"]
            class StopAfterTwo:
                def __len__(self):
                    return len(original_loader)
                def __iter__(self):
                    it = iter(original_loader)
                    yield next(it)
                    yield next(it)
                    raise RuntimeError("SIMULATED_RUNTIME_INTERRUPTION")
            kwargs["dataloader"] = StopAfterTwo()
            return original_train_one_epoch(*args, **kwargs)

        train_module.train_one_epoch = interrupt_after_two_batches
        try:
            try:
                train_module.train_model(
                    interrupted_model, interrupted_loader, validation_loader,
                    torch.device("cpu"), interrupted_dir, num_epochs=2
                )
                raise AssertionError("Expected simulated interruption.")
            except RuntimeError as exc:
                assert "SIMULATED_RUNTIME_INTERRUPTION" in str(exc)
        finally:
            train_module.train_one_epoch = original_train_one_epoch

        latest = interrupted_dir / "latest_checkpoint.pt"
        assert latest.exists(), "Checkpoint was not written before simulated interruption."
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        assert checkpoint["global_step"] == 2
        assert checkpoint["sampler_state_dict"]["position"] == 8
        assert checkpoint["sampler_state_dict"]["permutation"] is not None

        # Resume with a fresh model/loader and the checkpoint's saved sampler/RNG state.
        resumed_model = ToyModel()
        resumed_loader = make_loader(dataset)
        resumed_result = train_module.train_model(
            resumed_model, resumed_loader, validation_loader,
            torch.device("cpu"), interrupted_dir, num_epochs=2,
            resume_from=str(latest)
        )

        # Compare final weights and completed-step counts to the uninterrupted run.
        assert tensor_state_equal(full_model.state_dict(), resumed_model.state_dict()), (
            "Final model weights differ between uninterrupted and resumed training."
        )
        assert full_result["global_step"] == resumed_result["global_step"]
        assert full_result["history"] == resumed_result["history"]
        print("PASS: interrupted/resumed training matches uninterrupted training exactly.")
        print("Global steps:", full_result["global_step"])
        print("Final model parameters and history are identical.")


if __name__ == "__main__":
    main()
