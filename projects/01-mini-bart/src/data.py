"""Dataset loading, tokenization, and PyTorch DataLoaders for Mini-BART."""

import json

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, DataCollatorForSeq2Seq


MODEL_NAME = "facebook/bart-large-cnn"
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 1


class JsonSummarizationDataset(Dataset):
    """Small JSON-backed dataset used for smoke tests."""

    def __init__(
        self,
        records,
        tokenizer,
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        inputs = self.tokenizer(
            record["article"],
            max_length=self.max_input_length,
            truncation=True,
        )
        targets = self.tokenizer(
            text_target=record["highlights"],
            max_length=self.max_target_length,
            truncation=True,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }


def _make_collator(tokenizer):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )


def create_json_dataloader(
    json_path,
    tokenizer,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    dataset = JsonSummarizationDataset(records=records, tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_make_collator(tokenizer),
        num_workers=num_workers,
    )


def load_bart_tokenizer(model_name=MODEL_NAME):
    """Load the tokenizer shared by the BART teacher and student."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def load_cnn_dailymail(include_test=False):
    """
    Load CNN/DailyMail 3.0.0.

    The test split is deliberately excluded by default so it remains held
    out for final evaluation. Set include_test=True only for that stage.
    """
    from datasets import DatasetDict, load_dataset

    splits = {
        "train": load_dataset(DATASET_NAME, DATASET_VERSION, split="train"),
        "validation": load_dataset(
            DATASET_NAME, DATASET_VERSION, split="validation"
        ),
    }
    if include_test:
        splits["test"] = load_dataset(
            DATASET_NAME, DATASET_VERSION, split="test"
        )
    return DatasetDict(splits)


def preprocess_function(
    examples,
    tokenizer,
    max_input_length=MAX_INPUT_LENGTH,
    max_target_length=MAX_TARGET_LENGTH,
):
    """Tokenize article inputs and reference summaries."""
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["highlights"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_dataset(
    dataset,
    tokenizer,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
):
    """
    Tokenize supplied splits.

    Train and validation are required. Test is optional; if absent, an
    empty test split is added for compatibility with existing notebooks.
    A sample limit of zero intentionally creates an empty split.
    """
    if "train" not in dataset or "validation" not in dataset:
        raise KeyError("Dataset must contain 'train' and 'validation' splits.")

    sample_limits = {
        "train": max_train_samples,
        "validation": max_val_samples,
        "test": max_test_samples,
    }
    tokenized_dataset = {}

    for split, split_data in dataset.items():
        limit = sample_limits.get(split)
        if limit is not None:
            if limit < 0:
                raise ValueError(f"{split} sample limit cannot be negative.")
            split_data = split_data.select(
                range(min(limit, len(split_data)))
            )

        tokenized_dataset[split] = split_data.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=split_data.column_names,
            desc=f"Tokenizing {split} split",
        )

    # Preserve the notebook's three-split interface without loading,
    # tokenizing, or evaluating any test examples.
    if "test" not in tokenized_dataset:
        tokenized_dataset["test"] = tokenized_dataset["validation"].select([])

    return tokenized_dataset


class StatefulRandomSampler(Sampler):
    """Random sampler that can resume from an exact sample position."""

    def __init__(self, data_source, generator=None):
        self.data_source = data_source
        self.generator = generator if generator is not None else torch.Generator()
        self.permutation = None
        self.position = 0
        self.epoch = 0

    def __iter__(self):
        if self.permutation is None:
            self.permutation = torch.randperm(
                len(self.data_source),
                generator=self.generator,
            ).tolist()
            self.position = 0

        while self.position < len(self.permutation):
            index = self.permutation[self.position]
            self.position += 1
            yield index

        self.permutation = None
        self.position = 0
        self.epoch += 1

    def __len__(self):
        if self.permutation is None:
            return len(self.data_source)
        return len(self.permutation) - self.position

    def state_dict(self):
        return {
            "generator_state": self.generator.get_state(),
            "permutation": self.permutation,
            "position": self.position,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        permutation = state["permutation"]
        position = int(state["position"])
        if permutation is not None:
            if len(permutation) != len(self.data_source):
                raise ValueError(
                    "Sampler checkpoint dataset size does not match the current dataset."
                )
            if not 0 <= position <= len(permutation):
                raise ValueError("Sampler checkpoint position is invalid.")
        self.generator.set_state(state["generator_state"])
        self.permutation = permutation
        self.position = position
        self.epoch = int(state["epoch"])


def create_dataloaders(
    tokenized_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_workers=0,
):
    """
    Create train/validation/test loaders.

    Returns a three-item tuple for backward compatibility. If no real test
    split is supplied, test_loader wraps an empty view of validation.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if "train" not in tokenized_dataset or "validation" not in tokenized_dataset:
        raise KeyError(
            "Tokenized dataset must contain 'train' and 'validation' splits."
        )

    collator = _make_collator(tokenizer)
    pin_memory = torch.cuda.is_available()

    train_sampler = StatefulRandomSampler(
        tokenized_dataset["train"],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(123),
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_dataset = tokenized_dataset.get(
        "test", tokenized_dataset["validation"].select([])
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
