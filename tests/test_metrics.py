import math

import pytest

from benchmarks.metrics import noul_metrics, similarity_metrics


def record(expected, predicted):
    return {"expected": expected, "result": {"score": predicted}}


def test_similarity_metrics_include_pearson_and_tie_aware_spearman():
    records = [record(1, 4), record(1, 4), record(2, 2), record(3, 1)]
    result = similarity_metrics(records)
    assert result["pearson"] == pytest.approx(-0.9864400504)
    assert result["spearman"] == pytest.approx(-1.0)


def test_similarity_metrics_are_undefined_for_fewer_than_two_records():
    assert similarity_metrics([]) == {"pearson": None, "spearman": None}
    assert similarity_metrics([record(1, 1)]) == {"pearson": None, "spearman": None}


def test_noul_metrics_treat_noul_as_probability_of_yes():
    records = [
        {"expected": 1, "result": {"noul": 0.9}},
        {"expected": 0, "result": {"noul": 0.2}},
        {"expected": 1, "result": {"noul": 0.4}},
    ]
    result = noul_metrics(records)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["mean_confidence"] == pytest.approx((0.9 + 0.8 + 0.6) / 3)
    assert result["negative_log_loss"] == pytest.approx((-math.log(0.9) - math.log(0.8) - math.log(0.4)) / 3)
    assert result["expected_calibration_error"] == pytest.approx(0.3)
