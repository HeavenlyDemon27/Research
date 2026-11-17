"""ROUGE and extractiveness metrics (implementation to follow)."""
import re
# --------------------------------------------------
# 1. Text preprocessing
# --------------------------------------------------
def tokenize(text):
    """
    Convert text into lowercase word tokens.
    Punctuation is excluded.
    """
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens, n):
    """
    Generate consecutive n-grams from a list of tokens.
    """
    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]
# --------------------------------------------------
# 2. Source overlap / extractiveness
# --------------------------------------------------

def source_overlap(source, summary, n):
    """
    Calculate the proportion of summary n-grams
    that also occur in the source article.
    """
    source_tokens = tokenize(source)
    summary_tokens = tokenize(summary)

    source_ngrams = set(get_ngrams(source_tokens, n))
    summary_ngrams = get_ngrams(summary_tokens, n)

    if not summary_ngrams:
        return 0.0

    copied = sum(
        ngram in source_ngrams
        for ngram in summary_ngrams
    )

    return copied / len(summary_ngrams)


def compute_source_overlap(sources, summaries, n_values=(1, 2, 4)):
    """
    Calculate average copy and novel rates
    across multiple source-summary pairs.
    """
    if len(sources) != len(summaries):
        raise ValueError("Sources and summaries must have equal lengths.")

    if not sources:
        raise ValueError("At least one source-summary pair is required.")

    results = {}

    for n in n_values:
        rates = [
            source_overlap(source, summary, n)
            for source, summary in zip(sources, summaries)
        ]

        avg_copy = sum(rates) / len(rates)

        results[f"{n}-gram"] = {
            "copy_rate": avg_copy,
            "novel_rate": 1.0 - avg_copy
        }

    return results

# --------------------------------------------------
# 3. ROUGE evaluation
# --------------------------------------------------
def compute_rouge(predictions, references, use_stemmer=True):
    """
    Calculate ROUGE scores between generated summaries
    and human-written reference summaries.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal lengths.")

    if not predictions:
        raise ValueError("At least one prediction-reference pair is required.")

    import evaluate

    rouge = evaluate.load("rouge")

    return rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=use_stemmer
    )
