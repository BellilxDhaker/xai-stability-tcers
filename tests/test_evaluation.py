import numpy as np
import torch
from torch import nn

from src.config import ExperimentConfig
from src.evaluation.common import ExplanationMaps, METHODS
from src.evaluation.faithfulness import deletion_insertion
from src.evaluation.runner import Variant, match_variants
from src.evaluation.sanity_checks import randomized_models
from src.models.resnet import PixelClassifier, build_resnet18


class ToyModel(nn.Module):
    def forward(self, images):
        important = images[:, 0, 0, 0] * 3
        return torch.stack([torch.zeros_like(important), important], dim=1)


def test_deletion_and_insertion_endpoints():
    images = torch.ones(2, 3, 4, 4)
    maps = torch.zeros(2, 1, 4, 4)
    maps[:, 0, 0, 0] = 1
    fraction, deletion, insertion = deletion_insertion(
        ToyModel(), images, maps, torch.ones(2, dtype=torch.long),
        torch.zeros_like(images), 4, 4, torch.Generator().manual_seed(1),
    )
    assert fraction[0] == 0 and fraction[-1] == 1
    np.testing.assert_allclose(deletion[0], insertion[-1])
    np.testing.assert_allclose(deletion[-1], insertion[0])
    assert np.all(insertion[1] > deletion[1])


def test_matching_uses_each_control_once_and_excludes_flipped_predictions():
    def variant(distance, prediction=0):
        return Variant("test", 0, torch.empty(0), torch.tensor([prediction]),
                       np.array([1.0]), np.array([distance]))

    classic = [variant(0.02), variant(0.05, prediction=1), variant(0.1)]
    latent = [variant(0.019), variant(0.018), variant(0.051)]
    matches, latent_matches = match_variants(torch.tensor([0]), classic, latent, caliper=0.01)
    assert matches[:, 0].tolist() == [0, -1, -1]
    assert latent_matches[:, 0].tolist() == [0, -1, -1]


def test_randomization_cascades_but_independent_restores_other_layers():
    torch.set_num_threads(2)
    model = PixelClassifier(build_resnet18(pretrained=False))
    original = model.backbone.fc.weight.detach().clone()
    cascade = randomized_models(model, "cascading", seed=3)
    _, _, first = next(cascade)
    first_logits = first.backbone.fc.weight.detach().clone()
    _, _, second = next(cascade)
    assert not torch.equal(original, first_logits)
    torch.testing.assert_close(second.backbone.fc.weight, first_logits)
    independent = randomized_models(model, "independent", seed=3)
    next(independent)
    _, _, second = next(independent)
    torch.testing.assert_close(second.backbone.fc.weight, original)
    assert not torch.equal(second.backbone.layer4[0].conv1.weight, model.backbone.layer4[0].conv1.weight)


def test_batched_explainers_cache_signed_maps(tmp_path):
    torch.set_num_threads(2)
    model = PixelClassifier(build_resnet18(pretrained=False)).eval()
    config = ExperimentConfig(output_root=tmp_path, quick=True, ig_steps=3, forward_batch_size=4,
                              occlusion_patch=16, occlusion_stride=16)
    explain = ExplanationMaps(model, config, "test-model")
    images = torch.rand(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    before = explain(images, targets, signed=True, random_seed=5)
    assert tuple(before) == METHODS
    assert len(list(tmp_path.rglob("*.npy"))) == 1
    after = explain(images, targets, signed=True, random_seed=5)
    for method in METHODS:
        assert before[method].shape == (2, 1, 32, 32)
        torch.testing.assert_close(before[method], after[method])
