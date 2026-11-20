import torch
import torch.nn.functional as F


def knowledge_distillation_loss(
    teacher_logits,
    student_logits,
    labels,
    temperature=2.0,
):
    """
    Calculate temperature-scaled KL divergence between teacher and student.

    Args:
        teacher_logits: Teacher predictions [batch, seq_len, vocab_size]
        student_logits: Student predictions [batch, seq_len, vocab_size]
        labels: Target token IDs [batch, seq_len]
                Padding positions are -100.
        temperature: Softening temperature.

    Returns:
        Scalar KD loss.
    """

    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and student logits must have identical shapes.")

    if labels.shape != student_logits.shape[:2]:
        raise ValueError("Labels must match batch and sequence dimensions.")

    # Use FP32 for numerical stability during KL calculation.
    teacher_logits = teacher_logits.detach().float()
    student_logits = student_logits.float()

    # Soften the probability distributions.
    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=-1,
    )

    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=-1,
    )

    # KL divergence for each token and vocabulary element.
    token_kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    )

    # Sum over vocabulary dimension.
    token_kl = token_kl.sum(dim=-1)

    # Ignore padding positions.
    valid_mask = (labels != -100).float()

    # Average only over valid target tokens.
    kd_loss = (
        (token_kl * valid_mask).sum()
        / valid_mask.sum().clamp(min=1.0)
    )

    # Compensate for temperature scaling.
    kd_loss = kd_loss * (temperature ** 2)

    return kd_loss


def combined_distillation_loss(
    teacher_logits,
    student_logits,
    labels,
    temperature=2.0,
    lambda_kd=0.5,
):
    """
    Calculate CE + weighted KD loss.

    L_total = L_CE + lambda_kd * L_KD
    """

    if lambda_kd < 0:
        raise ValueError("lambda_kd cannot be negative.")

    vocab_size = student_logits.size(-1)

    # Standard supervised cross-entropy.
    ce_loss = F.cross_entropy(
        student_logits.float().reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

    # Knowledge distillation loss.
    kd_loss = knowledge_distillation_loss(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        labels=labels,
        temperature=temperature,
    )

    # Combine both objectives.
    total_loss = ce_loss + lambda_kd * kd_loss

    return {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "kd_loss": kd_loss,
    }
