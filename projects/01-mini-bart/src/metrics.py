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
        if not isinstance(n, int) or n < 1:
            raise ValueError("Every n-gram size must be a positive integer.")

        document_rates = []
        total_ngrams = 0
        total_copied = 0
        empty_documents = 0

        for source, summary in zip(sources, summaries):
            source_tokens = tokenize(source)
            summary_tokens = tokenize(summary)

            source_ngrams = set(get_ngrams(source_tokens, n))
            summary_ngrams = get_ngrams(summary_tokens, n)

            if not summary_ngrams:
                empty_documents += 1
                continue

            copied = sum(
                gram in source_ngrams for gram in summary_ngrams
            )

            document_rates.append(copied / len(summary_ngrams))
            total_copied += copied
            total_ngrams += len(summary_ngrams)

        macro_copy = (
            sum(document_rates) / len(document_rates)
            if document_rates else None
        )

        micro_copy = (
            total_copied / total_ngrams
            if total_ngrams else None
        )

        results[f"{n}-gram"] = {
            "macro_copy_rate": macro_copy,
            "macro_novel_rate": (
                1 - macro_copy if macro_copy is not None else None
            ),
            "micro_copy_rate": micro_copy,
            "micro_novel_rate": (
                1 - micro_copy if micro_copy is not None else None
            ),
            "empty_summary_count": empty_documents,
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
def extractive_fragment_metrics(source, summary):
    """Greedy longest-match fragment coverage and density, at token level."""
    src, summ = tokenize(source), tokenize(summary)
    if not src or not summ:
        return {"coverage": 0.0, "density": 0.0, "fragment_count": 0}
    fragments = []
    cursor = 0
    while cursor < len(summ):
        longest = 0
        for start in range(len(src)):
            match = 0
            while (cursor + match < len(summ) and start + match < len(src)
                   and summ[cursor + match] == src[start + match]):
                match += 1
            longest = max(longest, match)
        if longest:
            fragments.append(longest)
            cursor += longest
        else:
            cursor += 1
    return {
        "coverage": sum(fragments) / len(summ),
        "density": sum(length * length for length in fragments) / len(summ),
        "fragment_count": len(fragments),
    }

def compression_ratio(source, summary):
    source_len, summary_len = len(tokenize(source)), len(tokenize(summary))
    return summary_len / source_len if source_len else 0.0