
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.train import create_optimizer_and_scheduler
from src.train_kd import (
    train_kd_one_epoch,
    validate_kd,
)


# --------------------------------------------------
# Dummy seq2seq model
# --------------------------------------------------

class TinySeq2Seq(nn.Module):
    """
    Lightweight model that mimics the interface expected
    by the E2 training loop.

    Input:
        input_ids
        attention_mask
        labels

    Output:
        object containing logits
    """

    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        # Create decoder-like inputs by shifting labels.
        decoder_input_ids = labels.clone()

        decoder_input_ids[decoder_input_ids == -100] = 0

        decoder_input_ids = torch.roll(
            decoder_input_ids,
            shifts=1,
            dims=1,
        )

        decoder_input_ids[:, 0] = 0

        # Embed shifted decoder tokens.
        hidden = self.embedding(decoder_input_ids)

        # Add a simple article representation.
        article_representation = self.embedding(input_ids).mean(dim=1)

        hidden = hidden + article_representation.unsqueeze(1)

        logits = self.projection(hidden)

        return SimpleNamespace(logits=logits)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dummy_models():
    torch.manual_seed(42)

    teacher = TinySeq2Seq()
    student = TinySeq2Seq()

    # Freeze teacher.
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    # Student remains trainable.
    student.train()

    return teacher, student


@pytest.fixture
def dummy_batch():
    return {
        "input_ids": torch.tensor([
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ]),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]),
        "labels": torch.tensor([
            [1, 2, 3, -100],
            [4, 5, 6, -100],
        ]),
    }


@pytest.fixture
def dummy_dataloader(dummy_batch):
    return [dummy_batch, dummy_batch]


# --------------------------------------------------
# Helper
# --------------------------------------------------

def create_training_components(student, num_steps=2):
    optimizer, scheduler = create_optimizer_and_scheduler(
        student,
        total_training_steps=num_steps,
    )

    scaler = torch.amp.GradScaler(
        "cpu",
        enabled=False,
    )

    return optimizer, scheduler, scaler


# --------------------------------------------------
# 1. Student weights update after training
# --------------------------------------------------

def test_student_weights_update(
    dummy_models,
    dummy_dataloader,
    dummy_batch,
    device,
):
    teacher, student = dummy_models

    before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
    }

    optimizer, scheduler, scaler = create_training_components(
        student,
        num_steps=len(dummy_dataloader),
    )

    metrics = train_kd_one_epoch(
        teacher=teacher,
        student=student,
        dataloader=dummy_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
    )

    changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in student.named_parameters()
    )

    assert changed
    assert metrics["global_step"] == len(dummy_dataloader)


# --------------------------------------------------
# 2. Teacher weights remain unchanged
# --------------------------------------------------

def test_teacher_weights_remain_unchanged(
    dummy_models,
    dummy_dataloader,
    device,
):
    teacher, student = dummy_models

    before = {
        name: parameter.detach().clone()
        for name, parameter in teacher.named_parameters()
    }

    optimizer, scheduler, scaler = create_training_components(
        student,
        num_steps=len(dummy_dataloader),
    )

    train_kd_one_epoch(
        teacher=teacher,
        student=student,
        dataloader=dummy_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
    )

    for name, parameter in teacher.named_parameters():
        assert torch.equal(
            before[name],
            parameter.detach(),
        )


# --------------------------------------------------
# 3. Teacher receives no gradients
# --------------------------------------------------

def test_teacher_receives_no_gradients(
    dummy_models,
    dummy_dataloader,
    device,
):
    teacher, student = dummy_models

    optimizer, scheduler, scaler = create_training_components(
        student,
        num_steps=len(dummy_dataloader),
    )

    train_kd_one_epoch(
        teacher=teacher,
        student=student,
        dataloader=dummy_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
    )

    assert all(
        parameter.grad is None
        for parameter in teacher.parameters()
    )


# --------------------------------------------------
# 4. Training metrics are returned and finite
# --------------------------------------------------

def test_training_metrics_are_valid(
    dummy_models,
    dummy_dataloader,
    device,
):
    teacher, student = dummy_models

    optimizer, scheduler, scaler = create_training_components(
        student,
        num_steps=len(dummy_dataloader),
    )

    metrics = train_kd_one_epoch(
        teacher=teacher,
        student=student,
        dataloader=dummy_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
    )

    assert "loss" in metrics
    assert "ce_loss" in metrics
    assert "kd_loss" in metrics
    assert "global_step" in metrics

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["ce_loss"]))
    assert torch.isfinite(torch.tensor(metrics["kd_loss"]))

    assert metrics["global_step"] == len(dummy_dataloader)


# --------------------------------------------------
# 5. Validation returns finite losses
# --------------------------------------------------

def test_validation_metrics_are_finite(
    dummy_models,
    dummy_dataloader,
    device,
):
    teacher, student = dummy_models

    metrics = validate_kd(
        teacher=teacher,
        student=student,
        dataloader=dummy_dataloader,
        device=device,
    )

    assert "loss" in metrics
    assert "ce_loss" in metrics
    assert "kd_loss" in metrics

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["ce_loss"]))
    assert torch.isfinite(torch.tensor(metrics["kd_loss"]))


# --------------------------------------------------
# 6. Empty training loader raises an error
# --------------------------------------------------

def test_empty_training_dataloader_raises_error(
    dummy_models,
    device,
):
    teacher, student = dummy_models

    empty_dataloader = []

    optimizer, scheduler, scaler = create_training_components(
        student,
        num_steps=1,
    )

    with pytest.raises(ValueError, match="empty"):
        train_kd_one_epoch(
            teacher=teacher,
            student=student,
            dataloader=empty_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
        )


# --------------------------------------------------
# 7. Empty validation loader raises an error
# --------------------------------------------------

def test_empty_validation_dataloader_raises_error(
    dummy_models,
    device,
):
    teacher, student = dummy_models

    with pytest.raises(ValueError, match="empty"):
        validate_kd(
            teacher=teacher,
            student=student,
            dataloader=[],
            device=device,
        )


# --------------------------------------------------
# 8. Student receives gradients
# --------------------------------------------------

def test_student_receives_gradients(
    dummy_models,
    dummy_dataloader,
    device,
):
    teacher, student = dummy_models

    optimizer, scheduler, scaler = create_training_components(
        student,
        num_steps=len(dummy_dataloader),
    )

    train_kd_one_epoch(
        teacher=teacher,
        student=student,
        dataloader=dummy_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
    )

    # The student should be trainable and have gradients
    # from the final training batch.
    assert any(
        parameter.grad is not None
        for parameter in student.parameters()
    )
