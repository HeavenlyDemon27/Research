import pytest
import torch
import torch.nn.functional as F

from src.distillation import (
    knowledge_distillation_loss,
    combined_distillation_loss,
)


# --------------------------------------------------
# 1. Identical teacher and student -> KD loss is zero
# --------------------------------------------------

def test_identical_logits_produce_zero_kd_loss():
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([
        [1, 2, 3, -100],
        [4, 5, -100, -100],
    ])

    loss = knowledge_distillation_loss(
        teacher_logits=logits,
        student_logits=logits.clone(),
        labels=labels,
        temperature=2.0,
    )

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


# --------------------------------------------------
# 2. Padding positions must not affect KD loss
# --------------------------------------------------

def test_padding_positions_are_ignored():
    teacher_logits = torch.randn(2, 4, 10)
    student_logits = torch.randn(2, 4, 10)

    labels = torch.tensor([
        [1, 2, -100, -100],
        [3, 4, 5, -100],
    ])

    original_loss = knowledge_distillation_loss(
        teacher_logits,
        student_logits,
        labels,
        temperature=2.0,
    )

    # Change logits only at ignored/padded positions.
    teacher_modified = teacher_logits.clone()
    student_modified = student_logits.clone()

    teacher_modified[labels == -100] = 1000.0
    student_modified[labels == -100] = -1000.0

    modified_loss = knowledge_distillation_loss(
        teacher_modified,
        student_modified,
        labels,
        temperature=2.0,
    )

    assert torch.isclose(
        original_loss,
        modified_loss,
        atol=1e-6,
    )


# --------------------------------------------------
# 3. Student must receive gradients
# --------------------------------------------------

def test_student_receives_gradients():
    teacher_logits = torch.randn(2, 4, 10)

    student_logits = torch.randn(
        2, 4, 10,
        requires_grad=True,
    )

    labels = torch.tensor([
        [1, 2, 3, -100],
        [4, 5, 6, -100],
    ])

    loss = knowledge_distillation_loss(
        teacher_logits,
        student_logits,
        labels,
        temperature=2.0,
    )

    loss.backward()

    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert student_logits.grad.abs().sum() > 0


# --------------------------------------------------
# 4. Teacher must not receive gradients
# --------------------------------------------------

def test_teacher_does_not_receive_gradients():
    teacher_logits = torch.randn(
        2, 4, 10,
        requires_grad=True,
    )

    student_logits = torch.randn(
        2, 4, 10,
        requires_grad=True,
    )

    labels = torch.tensor([
        [1, 2, 3, -100],
        [4, 5, 6, -100],
    ])

    loss = knowledge_distillation_loss(
        teacher_logits,
        student_logits,
        labels,
        temperature=2.0,
    )

    loss.backward()

    assert teacher_logits.grad is None
    assert student_logits.grad is not None


# --------------------------------------------------
# 5. Invalid temperatures must raise errors
# --------------------------------------------------

@pytest.mark.parametrize("temperature", [0.0, -1.0, -5.0])
def test_invalid_temperature_raises_error(temperature):
    logits = torch.randn(2, 4, 10)
    labels = torch.ones(2, 4, dtype=torch.long)

    with pytest.raises(ValueError):
        knowledge_distillation_loss(
            logits,
            logits,
            labels,
            temperature=temperature,
        )


# --------------------------------------------------
# 6. Teacher and student shapes must match
# --------------------------------------------------

def test_mismatched_logits_shapes_raise_error():
    teacher_logits = torch.randn(2, 4, 10)
    student_logits = torch.randn(2, 4, 12)
    labels = torch.ones(2, 4, dtype=torch.long)

    with pytest.raises(ValueError):
        knowledge_distillation_loss(
            teacher_logits,
            student_logits,
            labels,
        )


# --------------------------------------------------
# 7. Labels must match batch and sequence dimensions
# --------------------------------------------------

def test_mismatched_label_shape_raises_error():
    teacher_logits = torch.randn(2, 4, 10)
    student_logits = torch.randn(2, 4, 10)
    labels = torch.ones(2, 3, dtype=torch.long)

    with pytest.raises(ValueError):
        knowledge_distillation_loss(
            teacher_logits,
            student_logits,
            labels,
        )


# --------------------------------------------------
# 8. Combined loss must equal CE + lambda * KD
# --------------------------------------------------

def test_combined_loss_matches_expected_formula():
    teacher_logits = torch.randn(2, 4, 10)
    student_logits = torch.randn(2, 4, 10)

    labels = torch.tensor([
        [1, 2, 3, -100],
        [4, 5, 6, -100],
    ])

    temperature = 2.0
    lambda_kd = 0.5

    result = combined_distillation_loss(
        teacher_logits,
        student_logits,
        labels,
        temperature=temperature,
        lambda_kd=lambda_kd,
    )

    expected_ce = F.cross_entropy(
        student_logits.reshape(-1, 10),
        labels.reshape(-1),
        ignore_index=-100,
    )

    expected_kd = knowledge_distillation_loss(
        teacher_logits,
        student_logits,
        labels,
        temperature=temperature,
    )

    expected_total = expected_ce + lambda_kd * expected_kd

    assert torch.isclose(
        result["ce_loss"],
        expected_ce,
        atol=1e-6,
    )

    assert torch.isclose(
        result["kd_loss"],
        expected_kd,
        atol=1e-6,
    )

    assert torch.isclose(
        result["loss"],
        expected_total,
        atol=1e-6,
    )


# --------------------------------------------------
# 9. Negative lambda must raise an error
# --------------------------------------------------

def test_negative_lambda_raises_error():
    logits = torch.randn(2, 4, 10)
    labels = torch.ones(2, 4, dtype=torch.long)

    with pytest.raises(ValueError):
        combined_distillation_loss(
            logits,
            logits,
            labels,
            lambda_kd=-0.1,
        )


# --------------------------------------------------
# 10. All losses must be finite
# --------------------------------------------------

def test_all_losses_are_finite():
    teacher_logits = torch.randn(2, 4, 10)
    student_logits = torch.randn(2, 4, 10)

    labels = torch.tensor([
        [1, 2, 3, -100],
        [4, 5, 6, -100],
    ])

    result = combined_distillation_loss(
        teacher_logits,
        student_logits,
        labels,
        temperature=2.0,
        lambda_kd=0.5,
    )

    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["ce_loss"])
    assert torch.isfinite(result["kd_loss"])