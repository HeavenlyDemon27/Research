
import json
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.model import (
    build_student_model,
    initialize_student_from_teacher,
)

MODEL_NAME = "facebook/bart-large-cnn"

# 1. Load tokenizer and teacher
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

teacher = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 2. Build and initialize student
student = build_student_model(teacher)

student = initialize_student_from_teacher(
    teacher,
    student
)

student.eval()

# 3. Load one example from the smoke-test dataset
with open("data/smoke_test.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

article = examples[0]["article"]
reference_summary = examples[0]["highlights"]

# 4. Tokenize article
inputs = tokenizer(
    article,
    return_tensors="pt",
    truncation=True,
    max_length=1024
)

# 5. Tokenize reference summary
targets = tokenizer(
    text_target=reference_summary,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

labels = targets["input_ids"]

# 6. Run student forward pass
with torch.no_grad():
    outputs = student(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=labels
    )

# 7. Inspect output
print("\n--- Input Information ---")
print("Input IDs shape:", inputs["input_ids"].shape)
print("Attention mask shape:", inputs["attention_mask"].shape)
print("Labels shape:", labels.shape)

print("\n--- Student Output ---")
print("Logits shape:", outputs.logits.shape)
print("Loss:", outputs.loss.item())

# 8. Verify output
assert outputs.logits.shape == (
    1,
    labels.shape[1],
    student.config.vocab_size
)

assert outputs.loss.ndim == 0
assert torch.isfinite(outputs.loss)

print("\nStudent forward-pass verification passed!")
