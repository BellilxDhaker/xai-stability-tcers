import numpy as np
import pytest
import torch

from src.metrics.stability import prediction_consistency, similarity_metrics, tcers
from src.perturbations.transforms import Perturbation


def test_geometry_warps_reference_and_excludes_rotation_fill():
    reference = torch.zeros(1, 1, 64, 64)
    reference[:, :, 10:30, 30:38] = 1
    rotation = Perturbation("rotation", 15)
    aligned, valid = rotation.align(reference)
    assert not valid[0, 0] and valid[32, 32]
    candidate = rotation.apply(reference)
    candidate[:, :, ~valid] = 100
    good = similarity_metrics(aligned[0, 0].numpy(), candidate[0, 0].numpy(), mask=valid.numpy())
    bad = similarity_metrics(reference[0, 0].numpy(), candidate[0, 0].numpy(), mask=valid.numpy())
    assert good["ssim"] == pytest.approx(1)
    assert bad["ssim"] < good["ssim"]
    resize = Perturbation("resize", 32)
    aligned_resize, mask = resize.align(reference)
    assert mask is None
    torch.testing.assert_close(aligned_resize, resize.apply(reference))


@pytest.mark.parametrize("name,parameter", [("brightness", 1.0), ("noise", 0.0)])
def test_identity_and_pixel_bounds(name, parameter):
    images = torch.rand(3, 3, 32, 32)
    torch.testing.assert_close(Perturbation(name, parameter).apply(images), images, rtol=0, atol=0)
    changed = Perturbation(name, 10.0).apply(images)
    assert changed.min() >= 0 and changed.max() <= 1


def test_metric_ranges_and_constant_control():
    image = np.eye(8)
    assert all(value == pytest.approx(1) for value in similarity_metrics(image, image).values())
    inverted = similarity_metrics(image, 1 - image)
    assert inverted["spearman"] == pytest.approx(-1)
    assert -1 <= inverted["ssim"] <= 1
    assert 0 <= inverted["cosine"] <= 1
    assert 0 <= inverted["iou"] <= 1
    constant = np.ones_like(image)
    assert tcers([similarity_metrics(constant, constant)])["tcers"] == pytest.approx(1)


def test_tcers_rescaling_and_missing_pairs():
    assert np.isnan(tcers([], total_transformations=2)["tcers"])
    metrics = {"ssim": 1.0, "spearman": 0.5, "cosine": 0.75}
    result = tcers([metrics], total_transformations=3)
    assert result["tcers"] == pytest.approx((1 + 0.75 + 0.75) / 3)
    assert result["excluded_transformations"] == 2
    assert tcers([{"ssim": -1, "spearman": -1, "cosine": 0}])["tcers"] == 0
    assert np.isnan(tcers([metrics], min_valid=2)["tcers"])
    with pytest.raises(ValueError):
        tcers([metrics], weights=(-1, 1, 1))


def test_prediction_consistency():
    assert prediction_consistency([0, 1, 1], [0, 0, 1]) == pytest.approx(2 / 3)
