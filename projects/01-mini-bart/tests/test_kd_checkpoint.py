
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from src.data import StatefulRandomSampler
from src.train import create_optimizer_and_scheduler

from src.train_kd import (
    capture_rng_state,
    get_kd_config,
    save_kd_checkpoint,
    load_kd_checkpoint,
    train_kd_one_epoch,
)


# ============================================================
# 1. TINY DUMMY MODELS AND DATASET
# ============================================================

class TinySeq2Seq(nn.Module):

    def __init__(self, vocab_size=24, hidden_size=12):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = 0

        decoder_input_ids = torch.roll(
            decoder_input_ids,
            shifts=1,
            dims=1,
        )
        decoder_input_ids[:, 0] = 0

        decoder_hidden = self.embedding(decoder_input_ids)

        encoder_hidden = self.embedding(input_ids).mean(dim=1)

        hidden = decoder_hidden + encoder_hidden.unsqueeze(1)
        hidden = self.dropout(hidden)

        logits = self.projection(hidden)

        return SimpleNamespace(logits=logits)


class TinyDataset(Dataset):

    def __init__(self, size=8):
        self.examples = []

        for index in range(size):
            self.examples.append({
                "input_ids": torch.tensor(
                    [
                        1 + index % 10,
                        2 + index % 10,
                        3 + index % 10,
                        4 + index % 10,
                    ],
                    dtype=torch.long,
                ),
                "attention_mask": torch.ones(4, dtype=torch.long),
                "labels": torch.tensor(
                    [
                        2 + index % 10,
                        3 + index % 10,
                        4 + index % 10,
                        -100,
                    ],
                    dtype=torch.long,
                ),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class RecordingScaler:
    """Small scaler stand-in to verify scaler state serialization."""

    def __init__(self):
        self.state = {"scale": 128.0, "growth_tracker": 7}

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


class StopAfterCheckpoint(Exception):
    pass


# ============================================================
# 2. COMMON HELPERS
# ============================================================

def create_components(seed=1234, dataset_size=8, batch_size=2):

    torch.manual_seed(seed)

    teacher = TinySeq2Seq()
    student = TinySeq2Seq()

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    dataset = TinyDataset(size=dataset_size)

    sampler = StatefulRandomSampler(
        dataset,
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        generator=torch.Generator().manual_seed(123),
    )

    optimizer, scheduler = create_optimizer_and_scheduler(
        student,
        total_training_steps=len(train_loader),
    )

    scaler = torch.amp.GradScaler(
        "cpu",
        enabled=False,
    )

    return (
        teacher,
        student,
        dataset,
        train_loader,
        optimizer,
        scheduler,
        scaler,
    )


def create_config():
    return get_kd_config(
        num_epochs=1,
        temperature=2.0,
        lambda_kd=0.5,
        gradient_accumulation_steps=1,
    )


def save_test_checkpoint(
    student,
    optimizer,
    scheduler,
    scaler,
    sampler,
    output_dir,
    config=None,
):
    if config is None:
        config = create_config()

    return save_kd_checkpoint(
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        gradient_accumulation_steps=1,
        train_sampler=sampler,
        epoch=0,
        global_step=0,
        best_val_loss=float("inf"),
        output_dir=str(output_dir),
        filename="test_kd_checkpoint.pt",
        config=config,
        include_training_state=True,
    )


def assert_nested_equal(actual, expected):

    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
        return

    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()

        for key in actual:
            assert_nested_equal(actual[key], expected[key])

        return

    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)

        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)

        return

    assert actual == expected


def create_optimizer_state(student, optimizer, scheduler):

    optimizer.zero_grad(set_to_none=True)

    loss = sum(
        parameter.square().mean()
        for parameter in student.parameters()
    )

    loss.backward()
    optimizer.step()
    scheduler.step()


# ============================================================
# 3. CHECKPOINT CREATION AND FILE INTEGRITY
# ============================================================

def test_checkpoint_creation_and_file_integrity(tmp_path):

    (
        teacher,
        student,
        dataset,
        loader,
        optimizer,
        scheduler,
        scaler,
    ) = create_components()

    checkpoint_path = save_test_checkpoint(
        student,
        optimizer,
        scheduler,
        scaler,
        loader.sampler,
        tmp_path,
    )

    assert os.path.exists(checkpoint_path)
    assert os.path.getsize(checkpoint_path) > 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = [
        "student_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "sampler_state_dict",
        "rng_state",
        "epoch",
        "global_step",
        "best_val_loss",
        "config",
    ]

    for key in required_keys:
        assert key in checkpoint

    assert checkpoint["epoch"] == 0
    assert checkpoint["global_step"] == 0

    # The atomic temporary file should not remain.
    assert not os.path.exists(checkpoint_path + ".tmp")


# ============================================================
# 4. STUDENT WEIGHT RESTORATION
# ============================================================

def test_student_weights_are_restored(tmp_path):

    (
        teacher,
        student,
        dataset,
        loader,
        optimizer,
        scheduler,
        scaler,
    ) = create_components()

    original_weights = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
    }

    checkpoint_path = save_test_checkpoint(
        student,
        optimizer,
        scheduler,
        scaler,
        loader.sampler,
        tmp_path,
    )

    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(10.0)

    load_kd_checkpoint(
        checkpoint_path=checkpoint_path,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        gradient_accumulation_steps=1,
        train_sampler=loader.sampler,
        device=torch.device("cpu"),
        expected_config=create_config(),
    )

    for name, parameter in student.named_parameters():
        assert torch.equal(
            parameter.detach(),
            original_weights[name],
        )


# ============================================================
# 5. OPTIMIZER, SCHEDULER, SCALER AND SAMPLER RESTORATION
# ============================================================

def test_all_training_states_are_restored(tmp_path):

    (
        teacher,
        student,
        dataset,
        loader,
        optimizer,
        scheduler,
        _,
    ) = create_components()

    scaler = RecordingScaler()

    # Create non-empty optimizer state and advance the scheduler.
    create_optimizer_state(student, optimizer, scheduler)

    # Consume one batch so that the sampler has a meaningful
    # mid-epoch permutation and position.
    iterator = iter(loader)
    next(iterator)

    assert loader.sampler.permutation is not None
    assert loader.sampler.position == 2

    checkpoint_path = save_test_checkpoint(
        student,
        optimizer,
        scheduler,
        scaler,
        loader.sampler,
        tmp_path,
    )

    saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
    saved_scheduler_state = copy.deepcopy(scheduler.state_dict())
    saved_scaler_state = copy.deepcopy(scaler.state_dict())
    saved_sampler_state = copy.deepcopy(loader.sampler.state_dict())

    # Deliberately alter the states.
    create_optimizer_state(student, optimizer, scheduler)

    scaler.state["scale"] = 1.0
    scaler.state["growth_tracker"] = 999

    next(iterator)

    # Restore.
    load_kd_checkpoint(
        checkpoint_path=checkpoint_path,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        gradient_accumulation_steps=1,
        train_sampler=loader.sampler,
        device=torch.device("cpu"),
        expected_config=create_config(),
    )

    assert_nested_equal(
        optimizer.state_dict(),
        saved_optimizer_state,
    )

    assert_nested_equal(
        scheduler.state_dict(),
        saved_scheduler_state,
    )

    assert_nested_equal(
        scaler.state_dict(),
        saved_scaler_state,
    )

    assert_nested_equal(
        loader.sampler.state_dict(),
        saved_sampler_state,
    )

    assert loader.sampler.position == 2


# ============================================================
# 6. RNG STATE RESTORATION
# ============================================================

def test_python_numpy_and_torch_rng_states_are_restored(tmp_path):

    (
        teacher,
        student,
        dataset,
        loader,
        optimizer,
        scheduler,
        scaler,
    ) = create_components()

    random.seed(456)
    np.random.seed(456)
    torch.manual_seed(456)

    checkpoint_path = save_test_checkpoint(
        student,
        optimizer,
        scheduler,
        scaler,
        loader.sampler,
        tmp_path,
    )

    # These are the values expected immediately after restoration.
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(5)

    # Corrupt the current RNG states.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    load_kd_checkpoint(
        checkpoint_path=checkpoint_path,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        gradient_accumulation_steps=1,
        train_sampler=loader.sampler,
        device=torch.device("cpu"),
        expected_config=create_config(),
    )

    actual_python = random.random()
    actual_numpy = np.random.random()
    actual_torch = torch.rand(5)

    assert actual_python == expected_python
    assert actual_numpy == expected_numpy
    assert torch.equal(actual_torch, expected_torch)


# ============================================================
# 7. INCOMPATIBLE CONFIGURATION MUST BE REJECTED
# ============================================================

def test_incompatible_configuration_is_rejected(tmp_path):

    (
        teacher,
        student,
        dataset,
        loader,
        optimizer,
        scheduler,
        scaler,
    ) = create_components()

    checkpoint_path = save_test_checkpoint(
        student,
        optimizer,
        scheduler,
        scaler,
        loader.sampler,
        tmp_path,
    )

    incompatible_config = create_config()
    incompatible_config["temperature"] = 4.0

    with pytest.raises(ValueError, match="configuration"):

        load_kd_checkpoint(
            checkpoint_path=checkpoint_path,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradient_accumulation_steps=1,
            train_sampler=loader.sampler,
            device=torch.device("cpu"),
            expected_config=incompatible_config,
        )


# ============================================================
# 8. INTERRUPTED VS UNINTERRUPTED TRAINING
# ============================================================

def run_uninterrupted_training():

    (
        teacher,
        student,
        dataset,
        loader,
        optimizer,
        scheduler,
        scaler,
    ) = create_components(seed=2026)

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    train_kd_one_epoch(
        teacher=teacher,
        student=student,
        dataloader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        scaler=scaler,
        gradient_accumulation_steps=1,
        global_step=0,
        checkpoint_every_steps=0,
    )

    return {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
    }


def run_interrupted_and_resumed_training(tmp_path):

    (
        teacher,
        student,
        dataset,
        loader,
        optimizer,
        scheduler,
        scaler,
    ) = create_components(seed=2026)

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    config = create_config()

    checkpoint_path = str(
        tmp_path / "interrupted_kd_checkpoint.pt"
    )

    def checkpoint_and_interrupt(step):

        save_kd_checkpoint(
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradient_accumulation_steps=1,
            train_sampler=loader.sampler,
            epoch=0,
            global_step=step,
            best_val_loss=float("inf"),
            output_dir=str(tmp_path),
            filename="interrupted_kd_checkpoint.pt",
            config=config,
            include_training_state=True,
        )

        raise StopAfterCheckpoint()

    # Stop immediately after the second batch has been checkpointed.
    with pytest.raises(StopAfterCheckpoint):

        train_kd_one_epoch(
            teacher=teacher,
            student=student,
            dataloader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("cpu"),
            scaler=scaler,
            gradient_accumulation_steps=1,
            global_step=0,
            checkpoint_callback=checkpoint_and_interrupt,
            checkpoint_every_steps=2,
        )

    assert os.path.exists(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["global_step"] == 2
    assert checkpoint["sampler_state_dict"]["position"] == 4

    # Build a fresh training setup. The checkpoint will restore
    # the student, optimizer, scheduler, sampler and RNG state.
    (
        resumed_teacher,
        resumed_student,
        resumed_dataset,
        resumed_loader,
        resumed_optimizer,
        resumed_scheduler,
        resumed_scaler,
    ) = create_components(seed=2026)

    load_state = load_kd_checkpoint(
        checkpoint_path=checkpoint_path,
        student=resumed_student,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        gradient_accumulation_steps=1,
        train_sampler=resumed_loader.sampler,
        device=torch.device("cpu"),
        expected_config=config,
    )

    assert load_state["global_step"] == 2
    assert load_state["epoch"] == 0

    # Continue from the exact saved sampler position.
    train_kd_one_epoch(
        teacher=resumed_teacher,
        student=resumed_student,
        dataloader=resumed_loader,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device=torch.device("cpu"),
        scaler=resumed_scaler,
        gradient_accumulation_steps=1,
        global_step=load_state["global_step"],
        checkpoint_every_steps=0,
    )

    return {
        name: parameter.detach().clone()
        for name, parameter in resumed_student.named_parameters()
    }


def test_interrupted_training_matches_uninterrupted_training(tmp_path):

    uninterrupted_weights = run_uninterrupted_training()

    resumed_weights = run_interrupted_and_resumed_training(
        tmp_path
    )

    assert uninterrupted_weights.keys() == resumed_weights.keys()

    for name in uninterrupted_weights:

        torch.testing.assert_close(
            uninterrupted_weights[name],
            resumed_weights[name],
            rtol=1e-6,
            atol=1e-7,
            msg=f"Mismatch after resume in parameter: {name}",
        )
