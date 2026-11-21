"""Dataset loading and preprocessing (implementation to follow)."""
#  Important design decisions understand these choices:
# Parameter	                   Initial value	         Purpose
# Input max length	              512	                 Maximum article tokens
# Target max length	               128	                 Maximum summary tokens
# Batch size	                     1	                 Initial GPU-memory-friendly setting
# Padding	                   Dynamic	                 Pad sequences to batch length
# Label padding	                  -100	                 Ignore padded positions in loss

#A) ATTENTION MASK: 
#2 article with different len. Article 1: [12, 45, 78, 91] & Article 2: [23, 56]
#to batch them together we pad shorter sequence : Article 2: [23, 56,  1,  1]
#Attention mask become: Article 1: [1, 1, 1, 1]  Article 2: [1, 1, 0, 0], where 1=actual token and 0=padding token
#The model uses this mask to avoid attending to padding positions in the encoder input


#B)Why I use -100 labels for? 
#imp. pytorch concept, our L. function will use cross-entropy: L_ce= -logP(y_correct)
#we dont padding positions to contribute to loss, PyTorch's CrossEntropyLoss supports ignore_index=-100
#Therefore, our data collator pads labels with -100, and those positions are ignored when calculating the loss.

#C)DYNAMIC PADDING
#We don't want every example to be padded to 512 tokens regardless of its length
#Instead, we pad each batch only to the longest sequence in that batch. This saves GPU memory and computation


from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
import json
from torch.utils.data import Dataset, DataLoader, Sampler
import torch


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_NAME = "facebook/bart-large-cnn"
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 1
class JsonSummarizationDataset(Dataset):

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

        article = record["article"]
        summary = record["highlights"]

        inputs = self.tokenizer(
            article,
            max_length=self.max_input_length,
            truncation=True,
        )

        targets = self.tokenizer(
            text_target=summary,
            max_length=self.max_target_length,
            truncation=True,
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }

def create_json_dataloader(
    json_path,
    tokenizer,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
):
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    dataset = JsonSummarizationDataset(
        records=records,
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
        num_workers=num_workers,
    )

    return dataloader
# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

def load_bart_tokenizer(model_name=MODEL_NAME): 
 #We load the tokenizer belonging to facebook/bart-large-cnn, 
 #why?Because our teacher and student share the same vocabulary and token-ID mapping. 
 # This is also essential when I(or you,the viewer) implement KD later: both models must interpret token IDs consistently
    """
    Load the tokenizer associated with BART.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


# --------------------------------------------------
# 3. Load CNN/DailyMail
# --------------------------------------------------

def load_cnn_dailymail(): 
 #Loads three splits: train, value, validation/cross-validation.
 #Train	Update student weights
 #Validation	Monitor performance and select checkpoints
 #Test	Final evaluation

    """
    Load train, validation and test splits.
    """
    from datasets import load_dataset
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_VERSION
    )

    return dataset


# --------------------------------------------------
# 4. Tokenize articles and reference summaries
# --------------------------------------------------

def preprocess_function(
    examples,
    tokenizer,
    max_input_length=MAX_INPUT_LENGTH,
    max_target_length=MAX_TARGET_LENGTH,
):
  #This is where text becomes numbers. 
  # ex: [OG ARTICLE]The government announced a new policy on Monday -->[tokenizer]0 245 918 32(illustrive token IDs)-->[input tensor]A numerical sequence that BART can process.
  #We tokenize the article and summary separately.
  #The article becomes input_ids, while the reference summary becomes labels
    """
    Convert articles and summaries into token IDs.
    """

    # Tokenize input articles
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True,
    )

    # Tokenize reference summaries
    labels = tokenizer(
        text_target=examples["highlights"],
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# --------------------------------------------------
# 5. Prepare tokenized dataset
# --------------------------------------------------

def prepare_dataset(
    dataset,
    tokenizer,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
):
    """
    Tokenize all dataset splits.

    Optional sample limits are useful for debugging.
    """

    sample_limits = {
        "train": max_train_samples,
        "validation": max_val_samples,
        "test": max_test_samples,
    }

    tokenized_dataset = {}

    for split, data in dataset.items():

        # Optional subset for debugging
        limit = sample_limits[split]

        if limit is not None:
            data = data.select(
                range(min(limit, len(data)))
            )

        # Tokenize the split
        tokenized_dataset[split] = data.map(
            lambda examples: preprocess_function(
                examples,
                tokenizer
            ),
            batched=True,
            remove_columns=data.column_names,
            desc=f"Tokenizing {split} split",
        )

    return tokenized_dataset


# --------------------------------------------------
# 6. Create PyTorch DataLoaders
# --------------------------------------------------

class StatefulRandomSampler(Sampler):
    """
    Random sampler that preserves its permutation and current position,
    allowing training to resume partway through an epoch.
    """

    def __init__(self, data_source, generator=None):
        self.data_source = data_source
        self.generator = generator or torch.Generator()

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
        self.generator.set_state(state["generator_state"])
        self.permutation = state["permutation"]
        self.position = state["position"]
        self.epoch = state["epoch"]

def create_dataloaders(
    tokenized_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_workers=0,
):
 #This converts our tokenized datasets into batches of PyTorch tensors.
 #For instance, with batch size 2: input_IDS : [2,1024] labels:[2,128]
 #These are illustrative maximum dimensions. 
 #With dynamic padding, the actual sequence dimensions will depend on the longest example in each batch.
    """
    Convert tokenized datasets into PyTorch DataLoaders.
    """

    # Dynamically pad sequences within each batch.
    # Padded label positions receive -100.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    train_sampler = StatefulRandomSampler(
       tokenized_dataset["train"],
       generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(123),
    )

    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        tokenized_dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
