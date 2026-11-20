"""
Held-out CNN/DailyMail evaluation for E0 (teacher), E1 (CE student), and E2 (KD student).

Run from the Mini-BART project root:
    python src/evaluate.py --model E0
    python src/evaluate.py --model E1 --checkpoint experiments/E1/checkpoints/best_checkpoint.pt
    python src/evaluate.py --model E2 --checkpoint experiments/E2/checkpoints/best_kd_checkpoint.pt

The full test split is used by default. --limit is intended only for smoke/pilot runs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, BartForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    DATASET_NAME,
    DATASET_VERSION,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    load_bart_tokenizer,
    load_cnn_dailymail,
    preprocess_function,
)
from src.model import build_student_model, load_teacher_model  # noqa: E402
from src.metrics import (  # noqa: E402
    compute_rouge,
    compute_source_overlap,
    extractive_fragment_metrics,
    compression_ratio,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "evaluation.json"


def read_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint_weights(path, model, state_key):
    """Load the exact model-state key written by the E1/E2 trainers."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(path, map_location="cpu")

    if state_key not in checkpoint:
        raise KeyError(
            f"{path} does not contain '{state_key}'. "
            f"Available top-level keys: {list(checkpoint.keys())}"
        )
    model.load_state_dict(checkpoint[state_key], strict=True)
    return checkpoint


def build_evaluation_model(experiment, pretrained_model_name, checkpoint_path, device):
    """
    E0 loads the pretrained teacher directly.
    E1/E2 instantiate the 6+6 architecture from the teacher config and load
    the corresponding trainer checkpoint key.
    """
    teacher = load_teacher_model(pretrained_model_name)

    if experiment == "E0":
        model = teacher
    else:
        model = build_student_model(teacher)
        if not checkpoint_path:
            raise ValueError(f"--checkpoint is required for {experiment}.")

        state_key = "model_state_dict" if experiment == "E1" else "student_state_dict"
        load_checkpoint_weights(checkpoint_path, model, state_key)

        # Release the full teacher before moving the student to the accelerator.
        del teacher

    model.to(device)
    model.eval()
    return model


def build_test_loader(raw_test, tokenizer, batch_size, max_examples=None):
    """Tokenize only test, retaining raw rows separately for text metrics."""
    if max_examples is not None:
        raw_test = raw_test.select(range(min(max_examples, len(raw_test))))

    tokenized_test = raw_test.map(
        lambda examples: preprocess_function(
            examples,
            tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        ),
        batched=True,
        remove_columns=raw_test.column_names,
        desc="Tokenizing held-out test split",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    loader = DataLoader(
        tokenized_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return raw_test, loader


def evaluate_model(model_name, checkpoint_path, config, output_dir, max_examples=None):
    if model_name not in {"E0", "E1", "E2"}:
        raise ValueError("model must be one of E0, E1, E2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_bart_tokenizer(config["model_name"])

    print(f"Loading CNN/DailyMail {DATASET_VERSION} test split...")
    raw_dataset = load_cnn_dailymail()
    raw_test, test_loader = build_test_loader(
        raw_dataset["test"],
        tokenizer,
        batch_size=config["batch_size"],
        max_examples=max_examples,
    )

    model = build_evaluation_model(
        model_name,
        config["model_name"],
        checkpoint_path,
        device,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generation = config["generation"]
    predictions, references, sources, ids = [], [], [], []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{model_name.lower()}_predictions.jsonl"

    start_time = time.perf_counter()
    with open(predictions_path, "w", encoding="utf-8") as out_file:
        row_offset = 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(test_loader):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=generation["num_beams"],
                    max_new_tokens=generation["max_new_tokens"],
                    min_new_tokens=generation["min_new_tokens"],
                    length_penalty=generation["length_penalty"],
                    no_repeat_ngram_size=generation["no_repeat_ngram_size"],
                    do_sample=generation["do_sample"],
                    early_stopping=generation["early_stopping"],
                )
                decoded = tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                batch_rows = [
                    raw_test[index]
                    for index in range(row_offset, row_offset + len(decoded))
                ]
                for row, prediction in zip(batch_rows, decoded):
                    article = row["article"]
                    reference = row["highlights"]
                    example_id = str(row.get("id", row_offset))

                    record = {
                        "id": example_id,
                        "article": article,
                        "reference": reference,
                        "prediction": prediction,
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                    ids.append(example_id)
                    sources.append(article)
                    references.append(reference)
                    predictions.append(prediction)
                    row_offset += 1

                if (batch_index + 1) % 100 == 0:
                    print(f"{model_name}: evaluated {row_offset}/{len(raw_test)}")

    elapsed_seconds = time.perf_counter() - start_time
    count = len(predictions)
    if count == 0:
        raise RuntimeError("The test loader produced zero examples.")

    print(f"Computing metrics for {count} generated summaries...")
    rouge = compute_rouge(predictions, references)
    overlap = compute_source_overlap(sources, predictions, n_values=(1, 2, 4))

    fragment_rows = [
        extractive_fragment_metrics(source, summary)
        for source, summary in zip(sources, predictions)
    ]
    fragment_metrics = {
        key: sum(row[key] for row in fragment_rows) / count
        for key in ("coverage", "density", "fragment_count")
    }
    compression = sum(
        compression_ratio(source, summary)
        for source, summary in zip(sources, predictions)
    ) / count

    summary = {
        "model": model_name,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "dataset": f"{DATASET_NAME}:{DATASET_VERSION}",
        "split": "test",
        "num_examples": count,
        "generation_config": generation,
        "rouge": rouge,
        "source_overlap": overlap,
        "extractive_fragments": fragment_metrics,
        "mean_compression_ratio": compression,
        "runtime_seconds": elapsed_seconds,
        "examples_per_second": count / elapsed_seconds if elapsed_seconds else None,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda"
            else None
        ),
        "predictions_file": predictions_path.name,
    }

    summary_path = output_dir / f"{model_name.lower()}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved predictions: {predictions_path}")
    print(f"Saved metrics:     {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mini-BART experiments on CNN/DailyMail test.")
    parser.add_argument("--model", required=True, choices=["E0", "E1", "E2"])
    parser.add_argument("--checkpoint", default=None, help="E1/E2 best checkpoint path; omit for E0.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional subset for a pilot only; omit for full test.")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer.")

    config = read_config(args.config)
    output_dir = args.output_dir or config["output_dir"]
    evaluate_model(
        args.model,
        args.checkpoint,
        config,
        output_dir,
        max_examples=args.limit,
    )


if __name__ == "__main__":
    main()
