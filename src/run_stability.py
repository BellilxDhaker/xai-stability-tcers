from src.config import config_from_args
from src.data.oxford_pets import evaluation_loader
from src.evaluation.common import ExplanationMaps, classifier, setup, stage_signature
from src.evaluation.runner import run_stability
from src.perturbations.vae_perturbations import load_vae, vae_signatures
from src.utils import save_json


def run(config):
    target_device = setup(config, "stability")
    model, checkpoint_key = classifier(config, target_device)
    vaes = {seed: load_vae(config, target_device, seed) for seed in config.evaluation_seeds}
    maps = ExplanationMaps(model, config, checkpoint_key)
    rows, reconstructions = run_stability(config, model, vaes, evaluation_loader(config), maps, target_device)
    rows.to_csv(config.output_root / "stability_raw.csv", index=False)
    reconstructions.to_csv(config.output_root / "reconstructions.csv", index=False)
    save_json({**stage_signature(config, checkpoint_key), "vae_sha256": vae_signatures(config)},
              config.output_root / "signature_stability.json")


def main():
    run(config_from_args("Measure classic and VAE explanation stability."))


if __name__ == "__main__":
    main()
