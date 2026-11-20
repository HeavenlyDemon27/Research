"""Student architecture and teacher loading (implementation to follow)."""

import copy
import torch

from transformers import BartForConditionalGeneration
TEACHER_LAYER_INDICES = [0, 2, 4, 6, 8, 10]


def build_student_model(teacher_model):
    """
    Build a smaller BART student using the teacher's configuration.

    The student will have 6 encoder layers and 6 decoder layers,
    while retaining the teacher's other architectural dimensions.
    """

    # Copy the teacher's configuration
    student_config = copy.deepcopy(teacher_model.config)

    # Reduce the number of transformer layers
    student_config.encoder_layers = 6
    student_config.decoder_layers = 6

    # Initialize a student model with this configuration
    student_model = BartForConditionalGeneration(student_config)

    return student_model
def initialize_student_from_teacher(teacher_model, student_model):
    """
    Initialize the student using selected teacher weights.
    """

    with torch.no_grad():

        # 1. Shared token embeddings
        student_model.model.shared.load_state_dict(
            teacher_model.model.shared.state_dict()
        )

        # 2. Encoder positional embeddings
        student_model.model.encoder.embed_positions.load_state_dict(
            teacher_model.model.encoder.embed_positions.state_dict()
        )

        # 3. Encoder embedding LayerNorm
        student_model.model.encoder.layernorm_embedding.load_state_dict(
            teacher_model.model.encoder.layernorm_embedding.state_dict()
        )

        # 4. Decoder positional embeddings
        student_model.model.decoder.embed_positions.load_state_dict(
            teacher_model.model.decoder.embed_positions.state_dict()
        )

        # 5. Decoder embedding LayerNorm
        student_model.model.decoder.layernorm_embedding.load_state_dict(
            teacher_model.model.decoder.layernorm_embedding.state_dict()
        )

        # 6. Selected encoder layers
        for student_idx, teacher_idx in enumerate(TEACHER_LAYER_INDICES):
            student_model.model.encoder.layers[student_idx].load_state_dict(
                teacher_model.model.encoder.layers[teacher_idx].state_dict()
            )

        # 7. Selected decoder layers
        for student_idx, teacher_idx in enumerate(TEACHER_LAYER_INDICES):
            student_model.model.decoder.layers[student_idx].load_state_dict(
                teacher_model.model.decoder.layers[teacher_idx].state_dict()
            )

        # 8. Final logits bias
        student_model.final_logits_bias.copy_(
            teacher_model.final_logits_bias
        )

    # Preserve BART's shared embedding / LM head weight tying.
    student_model.tie_weights()

    return student_model
def load_teacher_model(model_name="facebook/bart-large-cnn"):
    """
    Load the pretrained BART teacher.

    The teacher is used only to generate soft targets during
    knowledge distillation.
    """

    teacher_model = BartForConditionalGeneration.from_pretrained(
        model_name
    )

    return teacher_model


def prepare_kd_models(teacher_model):
    """
    Prepare the frozen teacher and freshly initialized student for E2.

    Returns:
        teacher_model: Frozen pretrained BART teacher.
        student_model: Trainable Mini-BART student.
    """

    # 1. Freeze the teacher completely.
    teacher_model.eval()

    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    # 2. Create a fresh Mini-BART student.
    student_model = build_student_model(teacher_model)

    # 3. Initialize student using selected teacher weights.
    student_model = initialize_student_from_teacher(
        teacher_model,
        student_model,
    )

    # 4. Ensure every student parameter is trainable.
    for parameter in student_model.parameters():
        parameter.requires_grad = True

    # Student starts in training mode.
    student_model.train()

    return teacher_model, student_model