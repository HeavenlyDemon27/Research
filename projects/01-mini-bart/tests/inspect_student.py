
from transformers import AutoModelForSeq2SeqLM

from src.model import build_student_model

MODEL_NAME = "facebook/bart-large-cnn"

# Load teacher
teacher = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Build student
student = build_student_model(teacher)

# Inspect architecture
print("\n--- Student Architecture ---")

print("Encoder layers:", len(student.model.encoder.layers))
print("Decoder layers:", len(student.model.decoder.layers))

print("Hidden dimension:", student.config.d_model)
print("Attention heads:", student.config.encoder_attention_heads)

# Count parameters
teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

print("\n--- Parameter Comparison ---")

print(f"Teacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,}")

print(f"Parameter reduction: {(1 - student_params / teacher_params) * 100:.2f}%")

assert len(student.model.encoder.layers) == 6
assert len(student.model.decoder.layers) == 6
assert student.config.d_model == 1024

print("\nStudent architecture verification passed!")
