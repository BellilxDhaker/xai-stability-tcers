import pandas as pd
import torch

from src.config import config_from_args
from src.data.oxford_pets import evaluation_loader
from src.evaluation.common import ExplanationMaps, classifier, setup, stage_signature
from src.evaluation.faithfulness import evaluate_faithfulness
from src.perturbations.vae_perturbations import decode_at_classifier_size, load_vae, resize_for_vae, vae_signatures
from src.utils import save_json


def run(config):
    target_device = setup(config, "faithfulness")
    model, checkpoint_key = classifier(config, target_device)
    maps = ExplanationMaps(model, config, checkpoint_key)
    vaes = {seed: load_vae(config, target_device, seed) for seed in config.evaluation_seeds}
    auc_tables, curve_tables = [], []
    loader = evaluation_loader(config)
    for batch, (images, _, indices) in enumerate(loader, start=1):
        images = images.to(target_device)
        ids = tuple(indices.tolist())
        for seed, vae in vaes.items():
            with torch.no_grad():
                mu, _ = vae.encode(resize_for_vae(images, config.vae_size))
                decoded = decode_at_classifier_size(vae, mu, images.shape[-2:])
            for domain, pixels in (("raw", images), ("decoded", decoded)):
                auc, curves = evaluate_faithfulness(model, config, maps, pixels, ids, seed, domain)
                auc_tables.append(auc)
                curve_tables.append(curves)
        print(f"Faithfulness batch {batch}/{len(loader)}", flush=True)
    pd.concat(auc_tables, ignore_index=True).to_csv(config.output_root / "faithfulness_auc.csv", index=False)
    pd.concat(curve_tables, ignore_index=True).to_csv(config.output_root / "faithfulness_curves.csv", index=False)
    save_json({**stage_signature(config, checkpoint_key), "vae_sha256": vae_signatures(config)},
              config.output_root / "signature_faithfulness.json")


def main():
    run(config_from_args("Evaluate deletion and insertion faithfulness."))


if __name__ == "__main__":
    main()
