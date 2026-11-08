from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_history(history: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    history.plot(x="epoch", y=["train_loss", "validation_loss"], ax=axes[0], title="Loss")
    history.plot(x="epoch", y=["train_accuracy", "validation_accuracy"], ax=axes[1], title="Accuracy")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def plot_stability(results: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=results, x="transformation", y="ssim", hue="method", ax=axis)
    axis.set(title="Explanation stability by transformation", ylabel="SSIM")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)

