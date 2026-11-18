import torch
from transformers import AutoModelForSeq2SeqLM

from src.model import (
    build_student_model,
    initialize_student_from_teacher,
    TEACHER_LAYER_INDICES,
)

MODEL_NAME = "facebook/bart-large-cnn"

teacher = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

student = build_student_model(teacher)

student = initialize_student_from_teacher(teacher, student)


def verify_module(student_module, teacher_module):
    student_state = student_module.state_dict()
    teacher_state = teacher_module.state_dict()

    assert student_state.keys() == teacher_state.keys()

    for key in student_state:
        assert torch.equal(
            student_state[key],
            teacher_state[key]
        ), f"Mismatch found in: {key}"

    return True


print("\n--- Verifying Shared Components ---")

verify_module(student.model.shared, teacher.model.shared)

verify_module(
    student.model.encoder.embed_positions,
    teacher.model.encoder.embed_positions
)

verify_module(
    student.model.encoder.layernorm_embedding,
    teacher.model.encoder.layernorm_embedding
)

verify_module(
    student.model.decoder.embed_positions,
    teacher.model.decoder.embed_positions
)

verify_module(
    student.model.decoder.layernorm_embedding,
    teacher.model.decoder.layernorm_embedding
)

assert torch.equal(
    student.final_logits_bias,
    teacher.final_logits_bias
)

print("All shared components verified!")


print("\n--- Verifying Encoder Layers ---")

for student_idx, teacher_idx in enumerate(TEACHER_LAYER_INDICES):
    verify_module(
        student.model.encoder.layers[student_idx],
        teacher.model.encoder.layers[teacher_idx]
    )

    print(f"Student encoder layer {student_idx} == Teacher encoder layer {teacher_idx}")


print("\n--- Verifying Decoder Layers ---")

for student_idx, teacher_idx in enumerate(TEACHER_LAYER_INDICES):
    verify_module(
        student.model.decoder.layers[student_idx],
        teacher.model.decoder.layers[teacher_idx]
    )

    print(f"Student decoder layer {student_idx} == Teacher decoder layer {teacher_idx}")


print("\n--- Verifying Weight Tying ---")

assert (
    student.model.shared.weight.data_ptr()
    == student.lm_head.weight.data_ptr()
)

print("Student shared embeddings and LM head are tied!")


print("\nALL WEIGHT TRANSFER CHECKS PASSED!")