from src.config import config_from_args
from src.data.oxford_pets import evaluation_loader
from src.evaluation.common import classifier, setup, stage_signature
from src.evaluation.sanity_checks import run_sanity
from src.utils import save_json


def run(config):
    target_device = setup(config, "sanity")
    model, checkpoint_key = classifier(config, target_device)
    rows = run_sanity(config, model, checkpoint_key, evaluation_loader(config), target_device)
    rows.to_csv(config.output_root / "sanity_raw.csv", index=False)
    save_json(stage_signature(config, checkpoint_key), config.output_root / "signature_sanity.json")


def main():
    run(config_from_args("Reproduce cascading and independent parameter randomization."))


if __name__ == "__main__":
    main()
