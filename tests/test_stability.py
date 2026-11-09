import numpy as np
import pytest

from src.metrics.stability import prediction_consistency, similarity_metrics, tcers


def test_similarity_metrics_are_bounded():
    result = similarity_metrics(np.eye(8), np.eye(8))
    assert result["ssim"] == pytest.approx(1.0)
    assert result["iou"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["cosine"] == pytest.approx(1.0)


def test_tcers_requires_minimum_valid_transformations():
    assert np.isnan(tcers([], min_valid=1)["tcers"])
    result = tcers([{"ssim": 1.0, "spearman": 0.5, "cosine": 0.75}])
    assert result["tcers"] == pytest.approx(0.75)


def test_prediction_consistency():
    assert prediction_consistency([0, 1, 1], [0, 0, 1]) == pytest.approx(2 / 3)

