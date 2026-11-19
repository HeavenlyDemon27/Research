
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import os
import tempfile

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import StatefulRandomSampler


class IndexedDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return index


def collect_indices(loader):
    indices = []
    for batch in loader:
        indices.extend(batch.tolist())
    return indices


def main():
    dataset = IndexedDataset()

    # Original sampler and DataLoader
    original_sampler = StatefulRandomSampler(
        dataset,
        generator=torch.Generator().manual_seed(42),
    )

    original_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=original_sampler,
        num_workers=0,
        generator=torch.Generator().manual_seed(123),
    )

    original_iterator = iter(original_loader)

    # Consume the first two batches (8 examples)
    first_batch = next(original_iterator).tolist()
    second_batch = next(original_iterator).tolist()

    consumed = first_batch + second_batch

    assert original_sampler.position == 8
    print("PASS: First 8 examples consumed.")
    print("Consumed:", consumed)

    # Save sampler state
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = Path(f.name)

    try:
        torch.save(original_sampler.state_dict(), checkpoint_path)

        # Continue original training to obtain expected remaining order
        expected_remaining = []
        for batch in original_iterator:
            expected_remaining.extend(batch.tolist())

        assert len(consumed + expected_remaining) == 16
        assert len(set(consumed + expected_remaining)) == 16

        print("PASS: Original sampler completed its epoch.")

        # Create a fresh sampler and restore saved state
        restored_sampler = StatefulRandomSampler(
            dataset,
            generator=torch.Generator().manual_seed(999),
        )

        restored_sampler.load_state_dict(
            torch.load(checkpoint_path, weights_only=False)
        )

        restored_loader = DataLoader(
            dataset,
            batch_size=4,
            sampler=restored_sampler,
            num_workers=0,
            generator=torch.Generator().manual_seed(123),
        )

        actual_remaining = collect_indices(restored_loader)

        # Compare the resumed order with the original remaining order
        assert actual_remaining == expected_remaining, (
            f"Order mismatch!\n"
            f"Expected: {expected_remaining}\n"
            f"Actual:   {actual_remaining}"
        )

        print("PASS: Resumed sampler produces exactly the same remaining order.")
        print("Expected:", expected_remaining)
        print("Actual:  ", actual_remaining)

        # Verify complete epoch coverage
        complete_order = consumed + actual_remaining

        assert len(complete_order) == 16
        assert len(set(complete_order)) == 16

        print("PASS: All 16 examples appear exactly once.")
        print("\n========== MID-EPOCH RESUME TEST PASSED ==========")

    finally:
        checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
