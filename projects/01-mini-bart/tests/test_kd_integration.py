
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from src.data import StatefulRandomSampler
import src.train_kd as train_kd_module

from src.train_kd import train_kd_model


# ============================================================
# 1. DUMMY MODEL AND DATASET
# ============================================================

class TinySeq2Seq(nn.Module):

    def __init__(self, vocab_size=24, hidden_size=12):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
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


def create_loaders(dataset):

    sampler = StatefulRandomSampler(
        dataset,
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        generator=torch.Generator().manual_seed(123),
    )

    val_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


class SimulatedInterruption(Exception):
    pass


# ============================================================
# 2. END-TO-END TRAINING AND MID-EPOCH RESUME
# ============================================================

def test_e2_end_to_end_training_and_mid_epoch_resume(
    tmp_path,
    monkeypatch,
):

    torch.manual_seed(101)

    dataset = TinyDataset(size=8)

    train_loader, val_loader = create_loaders(dataset)

    teacher = TinySeq2Seq()
    student = TinySeq2Seq()

    original_teacher_weights = {
        name: parameter.detach().clone()
        for name, parameter in teacher.named_parameters()
    }

    

    original_train_function = train_kd_module.train_kd_one_epoch

    # --------------------------------------------------------
    # A. Simulate interruption after a real mid-epoch save
    # --------------------------------------------------------

    def interrupt_after_checkpoint(*args, **kwargs):

        original_callback = kwargs["checkpoint_callback"]

        def save_then_interrupt(step):
            original_callback(step)
            raise SimulatedInterruption()

        kwargs["checkpoint_callback"] = save_then_interrupt

        return original_train_function(*args, **kwargs)

    monkeypatch.setattr(
        train_kd_module,
        "train_kd_one_epoch",
        interrupt_after_checkpoint,
    )

    with pytest.raises(SimulatedInterruption):

        train_kd_model(
            teacher=teacher,
            student=student,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=torch.device("cpu"),
            output_dir=str(tmp_path),
            num_epochs=2,
            temperature=2.0,
            lambda_kd=0.5,
            checkpoint_every_steps=1,
        gradient_accumulation_steps=2,
            gradient_accumulation_steps=2,
        )

    latest_path = tmp_path / "latest_kd_checkpoint.pt"

    assert latest_path.exists()

    mid_epoch_checkpoint = torch.load(
        latest_path,
        map_location="cpu",
        weights_only=False,
    )

    assert mid_epoch_checkpoint["epoch"] == 0
    assert mid_epoch_checkpoint["global_step"] == 1

    sampler_state = mid_epoch_checkpoint["sampler_state_dict"]

    assert sampler_state["permutation"] is not None
    assert sampler_state["position"] == 4

    # Teacher must already have been frozen by train_kd_model.
    assert all(
        not parameter.requires_grad
        for parameter in teacher.parameters()
    )

    # --------------------------------------------------------
    # B. Restore the original training function
    # --------------------------------------------------------

    monkeypatch.setattr(
        train_kd_module,
        "train_kd_one_epoch",
        original_train_function,
    )

    # Create a fresh student. Its weights will be overwritten
    # by the mid-epoch checkpoint during resume.
    resumed_student = TinySeq2Seq()

    resumed_train_loader, resumed_val_loader = create_loaders(
        dataset
    )

    # --------------------------------------------------------
    # C. Resume and finish both epochs
    # --------------------------------------------------------

    results = train_kd_model(
        teacher=teacher,
        student=resumed_student,
        train_dataloader=resumed_train_loader,
        val_dataloader=resumed_val_loader,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        num_epochs=2,
        temperature=2.0,
        lambda_kd=0.5,
        resume_from=str(latest_path),
        checkpoint_every_steps=1,
        gradient_accumulation_steps=2,
    )

    # --------------------------------------------------------
    # D. Verify training completed
    # --------------------------------------------------------

    assert results["global_step"] == 4
    assert len(results["history"]) == 2

    assert results["best_val_loss"] < float("inf")

    # --------------------------------------------------------
    # E. Verify best and latest checkpoints
    # --------------------------------------------------------

    best_path = tmp_path / "best_kd_checkpoint.pt"

    assert best_path.exists()
    assert latest_path.exists()

    best_checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )

    latest_checkpoint = torch.load(
        latest_path,
        map_location="cpu",
        weights_only=False,
    )

    assert "student_state_dict" in best_checkpoint
    assert "student_state_dict" in latest_checkpoint

    assert "optimizer_state_dict" in latest_checkpoint
    assert "scheduler_state_dict" in latest_checkpoint
    assert "scaler_state_dict" in latest_checkpoint
    assert "sampler_state_dict" in latest_checkpoint
    assert "rng_state" in latest_checkpoint

    # --------------------------------------------------------
    # F. Verify student weights actually changed after resume
    # --------------------------------------------------------

    checkpoint_student_weights = mid_epoch_checkpoint[
        "student_state_dict"
    ]

    student_changed = any(
        not torch.equal(
            checkpoint_student_weights[name],
            parameter.detach(),
        )
        for name, parameter in resumed_student.named_parameters()
    )

    assert student_changed

    # --------------------------------------------------------
    # G. Verify teacher stayed frozen and unchanged
    # --------------------------------------------------------

    assert all(
        not parameter.requires_grad
        for parameter in teacher.parameters()
    )

    for name, parameter in teacher.named_parameters():
        assert torch.equal(
            original_teacher_weights[name],
            parameter.detach(),
        )

        assert parameter.grad is None

    # --------------------------------------------------------
    # H. Verify CE, KD and total losses are finite
    # --------------------------------------------------------

    for epoch_result in results["history"]:

        for phase in ("train", "validation"):

            metrics = epoch_result[phase]

            for metric_name in ("loss", "ce_loss", "kd_loss"):

                value = metrics[metric_name]

                assert torch.isfinite(
                    torch.tensor(value)
                ), (
                    f"Non-finite {phase} {metric_name}"
                )

    # --------------------------------------------------------
    # I. Verify final checkpoint is at epoch boundary
    # --------------------------------------------------------

    assert latest_checkpoint["epoch"] == 2
    assert latest_checkpoint["global_step"] == 4
