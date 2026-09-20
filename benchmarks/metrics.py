"""Metrics shared by benchmark tasks."""

from __future__ import annotations

import math
from typing import Any


def classification_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {
            "accuracy": None,
            "mean_confidence": None,
            "negative_log_loss": None,
            "expected_calibration_error": None,
        }
    correct, confidences, losses = 0, [], []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for record in records:
        result = record["result"]
        probabilities = result["probabilities"]
        confidence = max(probabilities.values())
        hit = result["choice"] == record["expected"]
        correct += hit
        confidences.append(confidence)
        losses.append(-math.log(max(probabilities[record["expected"]], 1e-15)))
        bins[min(int(confidence * 10), 9)].append((confidence, hit))
    ece = sum(
        len(items) / len(records)
        * abs(sum(confidence for confidence, _ in items) / len(items) - sum(hit for _, hit in items) / len(items))
        for items in bins
        if items
    )
    return {
        "accuracy": correct / len(records),
        "mean_confidence": sum(confidences) / len(records),
        "negative_log_loss": sum(losses) / len(records),
        "expected_calibration_error": ece,
    }


def noul_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {
            "accuracy": None,
            "mean_confidence": None,
            "negative_log_loss": None,
            "expected_calibration_error": None,
        }
    correct, confidences, losses = 0, [], []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for record in records:
        probability = float(record["result"]["noul"])
        expected = bool(record["expected"])
        confidence = max(probability, 1 - probability)
        hit = (probability >= 0.5) == expected
        correct += hit
        confidences.append(confidence)
        losses.append(-math.log(max(probability if expected else 1 - probability, 1e-15)))
        bins[min(int(confidence * 10), 9)].append((confidence, hit))
    ece = sum(
        len(items) / len(records)
        * abs(sum(confidence for confidence, _ in items) / len(items) - sum(hit for _, hit in items) / len(items))
        for items in bins
        if items
    )
    return {
        "accuracy": correct / len(records),
        "mean_confidence": sum(confidences) / len(records),
        "negative_log_loss": sum(losses) / len(records),
        "expected_calibration_error": ece,
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end + 1) / 2
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def similarity_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    expected = [float(record["expected"]) for record in records]
    predicted = [float(record["result"]["score"]) for record in records]
    return {
        "pearson": _pearson(expected, predicted),
        "spearman": _pearson(_ranks(expected), _ranks(predicted)) if records else None,
    }
