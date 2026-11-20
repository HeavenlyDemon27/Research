import torch
from transformers import BartConfig, BartForConditionalGeneration

from src.model import prepare_kd_models, TEACHER_LAYER_INDICES


def create_dummy_teacher():
    """
    Create a small 12-layer BART teacher for testing.
    No pretrained weights or downloads required.
    """

    config = BartConfig(
        vocab_size=32,
        max_position_embeddings=32,
        encoder_layers=12,
        decoder_layers=12,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        d_model=32,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
    )

    return BartForConditionalGeneration(config)


def test_student_has_six_encoder_and_decoder_layers():
    teacher = create_dummy_teacher()

    teacher, student = prepare_kd_models(teacher)

    assert len(student.model.encoder.layers) == 6
    assert len(student.model.decoder.layers) == 6


def test_teacher_is_frozen():
    teacher = create_dummy_teacher()

    teacher, student = prepare_kd_models(teacher)

    assert all(
        not parameter.requires_grad
        for parameter in teacher.parameters()
    )


def test_student_is_trainable():
    teacher = create_dummy_teacher()

    teacher, student = prepare_kd_models(teacher)

    assert all(
        parameter.requires_grad
        for parameter in student.parameters()
    )


def test_teacher_is_in_eval_mode():
    teacher = create_dummy_teacher()

    teacher, student = prepare_kd_models(teacher)

    assert not teacher.training


def test_student_is_in_training_mode():
    teacher = create_dummy_teacher()

    teacher, student = prepare_kd_models(teacher)

    assert student.training


def test_selected_encoder_layers_are_copied():
    teacher = create_dummy_teacher()

    teacher, student = prepare_kd_models(teacher)

    for student_idx, teacher_idx in enumerate(TEACHER_LAYER_INDICES):
        teacher_layer = teacher.model.encoder.layers[teacher_idx]
        student_layer = student.model.encoder.layers[student_idx]

        for teacher_param, student_param in zip(
            teacher_layer.parameters(),
            student_layer.parameters(),
        ):
            assert torch.equal(teacher_param, student_param)


def test_selected_decoder_layers_are_copied():
    teacher = create_dummy_teacher()

    teacher, student = prepare_kd_models(teacher)

    for student_idx, teacher_idx in enumerate(TEACHER_LAYER_INDICES):
        teacher_layer = teacher.model.decoder.layers[teacher_idx]
        student_layer = student.model.decoder.layers[student_idx]

        for teacher_param, student_param in zip(
            teacher_layer.parameters(),
            student_layer.parameters(),
        ):
            assert torch.equal(teacher_param, student_param)


def test_student_and_teacher_parameters_are_independent():
    teacher = create_dummy_teacher()

    teacher, student = prepare_kd_models(teacher)

    teacher_param = next(teacher.parameters())
    student_param = next(student.parameters())

    assert teacher_param.data_ptr() != student_param.data_ptr()