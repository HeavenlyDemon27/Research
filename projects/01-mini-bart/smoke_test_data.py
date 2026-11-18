
import torch

from src.data import (
    load_bart_tokenizer,
    create_json_dataloader,
)

# 1. Load the tokenizer
tokenizer = load_bart_tokenizer()

# 2. Create the DataLoader
dataloader = create_json_dataloader(
    json_path="data/smoke_test.json",
    tokenizer=tokenizer,
    batch_size=2,
    shuffle=False,
)

# 3. Retrieve the first batch
batch = next(iter(dataloader))

# 4. Inspect the batch
print("\n--- Batch Information ---")

for key, value in batch.items():
    print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")

# 5. Inspect padding
print("\n--- Padding Information ---")

print("Input padding token ID:", tokenizer.pad_token_id)
print("Number of padded input tokens:",
      (batch["input_ids"] == tokenizer.pad_token_id).sum().item())

print("Number of ignored label tokens (-100):",
      (batch["labels"] == -100).sum().item())

# 6. Decode one example
print("\n--- Decoded Example ---")

article = tokenizer.decode(
    batch["input_ids"][0],
    skip_special_tokens=True,
)

summary = tokenizer.decode(
    batch["labels"][0][batch["labels"][0] != -100],
    skip_special_tokens=True,
)

print("Article:", article)
print("Reference summary:", summary)

# 7. Basic validation
assert batch["input_ids"].ndim == 2
assert batch["attention_mask"].shape == batch["input_ids"].shape
assert batch["labels"].ndim == 2
assert batch["input_ids"].shape[0] == 2

assert torch.all(
    (batch["attention_mask"] == 0) |
    (batch["attention_mask"] == 1)
)

print("\nAll data-pipeline checks passed!")
print("\n--- Full Dataset Validation ---")

total_examples = 0

for batch in dataloader:
    total_examples += batch["input_ids"].shape[0]

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["labels"].ndim == 2

print("Total examples processed:", total_examples)

assert total_examples == 3

print("Full dataset validation passed!")