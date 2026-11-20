from src.metrics import (
    tokenize, novel_ngram_ratio, extractive_fragment_metrics,
    compression_ratio, compute_text_metrics,
)

def test_tokenization_is_case_insensitive():
    assert tokenize("Hello hello!") == ["hello", "hello", "!"]

def test_novel_unigram_ratio():
    assert novel_ngram_ratio("cat sat", "cat dog", n=1) == 0.5

def test_novel_ngram_empty_summary():
    assert novel_ngram_ratio("source text", "", n=2) == 0.0

def test_fragment_metrics_full_copy():
    metrics = extractive_fragment_metrics("one two three", "one two three")
    assert metrics["coverage"] == 1.0
    assert metrics["density"] == 3.0
    assert metrics["fragment_count"] == 1

def test_fragment_metrics_partial_copy():
    metrics = extractive_fragment_metrics("alpha beta gamma", "alpha beta new")
    assert metrics["coverage"] == 2 / 3
    assert metrics["density"] == 4 / 3

def test_compression_ratio():
    assert compression_ratio("a b c d", "a b") == 0.5
    assert compression_ratio("", "anything") == 0.0

def test_metrics_keys():
    keys = set(compute_text_metrics("source", "summary"))
    assert keys == {"coverage", "density", "fragment_count", "novel_1gram_ratio",
                    "novel_2gram_ratio", "novel_3gram_ratio", "compression_ratio"}
