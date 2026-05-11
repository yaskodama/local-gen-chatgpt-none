from __future__ import annotations

import math
import re


def evaluate_text(generated: str, train_text: str, loss: float) -> dict[str, float]:
    unique_ratio = len(set(generated)) / max(1, len(generated))
    train_chars = set(train_text)
    valid_ratio = sum(1 for ch in generated if ch in train_chars) / max(1, len(generated))
    whitespace_ratio = sum(1 for ch in generated if ch.isspace()) / max(1, len(generated))
    repeated_penalty = max_repeated_char_run(generated) / max(1, len(generated))
    word_like = len(re.findall(r"[A-Za-z]{2,}|[ぁ-んァ-ン一-龥]{2,}", generated))
    ngram_score = ngram_overlap(generated, train_text, n=3)
    line_score = line_shape_score(generated)
    phrase_repeat_score = repeated_ngram_score(generated, n=4)
    copy_balance_score = copy_balance(ngram_overlap(generated, train_text, n=8))

    loss_score = 1.0 / (1.0 + max(loss, 0.0))
    diversity_score = min(unique_ratio * 8.0, 1.0)
    whitespace_score = max(0.0, 1.0 - abs(whitespace_ratio - 0.18) * 3.0)
    word_score = min(word_like / 8.0, 1.0)
    repeat_score = max(0.0, 1.0 - repeated_penalty * 8.0)

    score = (
        0.82 * loss_score
        + 0.03 * diversity_score
        + 0.03 * valid_ratio
        + 0.03 * whitespace_score
        + 0.02 * word_score
        + 0.02 * repeat_score
        + 0.02 * ngram_score
        + 0.01 * line_score
        + 0.015 * phrase_repeat_score
        + 0.005 * copy_balance_score
    )
    if math.isnan(score) or math.isinf(score):
        score = 0.0
    return {
        "score": score,
        "loss_score": loss_score,
        "diversity_score": diversity_score,
        "valid_ratio": valid_ratio,
        "whitespace_score": whitespace_score,
        "word_score": word_score,
        "repeat_score": repeat_score,
        "ngram_score": ngram_score,
        "line_score": line_score,
        "phrase_repeat_score": phrase_repeat_score,
        "copy_balance_score": copy_balance_score,
    }


def max_repeated_char_run(text: str) -> int:
    longest = 0
    current = 0
    prev = None
    for ch in text:
        current = current + 1 if ch == prev else 1
        longest = max(longest, current)
        prev = ch
    return longest


def ngram_overlap(generated: str, train_text: str, n: int = 3) -> float:
    if len(generated) < n or len(train_text) < n:
        return 0.0
    train_ngrams = {train_text[i : i + n] for i in range(len(train_text) - n + 1)}
    generated_ngrams = [generated[i : i + n] for i in range(len(generated) - n + 1)]
    if not generated_ngrams:
        return 0.0
    return sum(1 for item in generated_ngrams if item in train_ngrams) / len(generated_ngrams)


def line_shape_score(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    reasonable = sum(1 for line in lines if 8 <= len(line) <= 100)
    return reasonable / len(lines)


def repeated_ngram_score(text: str, n: int = 4) -> float:
    compact = re.sub(r"\s+", " ", text)
    if len(compact) < n:
        return 1.0
    ngrams = [compact[i : i + n] for i in range(len(compact) - n + 1)]
    if not ngrams:
        return 1.0
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return max(0.0, min(1.0, unique_ratio * 1.6))


def copy_balance(overlap: float) -> float:
    # Some overlap means the model learned the corpus; too much usually means memorization.
    return max(0.0, 1.0 - abs(overlap - 0.35) * 2.0)
