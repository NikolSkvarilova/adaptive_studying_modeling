from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_all_calibrations(calibrations: dict, file_path: Path) -> None:
    """Overlay calibrations for each model. Produces as many plots as there are models in the dictionary.

    Args:
        calibrations (dict): dictionary with models as keys, and calibrations as values.
        file_path (Path): directory for the plots.
    """
    file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)

    for model, lines in calibrations.items():
        plt.figure(figsize=(8, 5))

        all_x = []
        all_y = []

        # Plot individual calibrations
        for line in lines:
            plt.plot(
                line[0],
                line[1],
                color="lightgray",
                linewidth=1.5,
                alpha=0.8,
            )

            all_x.extend(line[0])
            all_y.extend(line[1])

        # Plot the mean
        if all_x and all_y:
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(all_x, bins)

            bin_means_x = []
            bin_means_y = []

            for i in range(1, len(bins)):
                mask = bin_indices == i
                if np.any(mask):
                    bin_means_x.append(np.mean(np.array(all_x)[mask]))
                    bin_means_y.append(np.mean(np.array(all_y)[mask]))

            plt.plot(
                bin_means_x,
                bin_means_y,
                color="#4e9b4e",
                linewidth=3,
                marker="o",
                label="Average Calibration",
                zorder=5,
            )

        # Plot the perfect calibration
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            label="Perfect calibration",
            zorder=4,
        )

        plt.title(f"Calibration Plot ({model})")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Actual Fraction of Correct Answers")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(file_path / f"{model}_calibration.png", dpi=300)
        plt.show()
        plt.close()


def plot_all_resolutions(resolutions: dict, file_path: Path) -> None:
    """Overlay resolutions for each model. Produces as many plots as there are models in the dictionary.

    Args:
        resolutions (dict): dictionary with models as keys, and resolutions as values.
        file_path (Path): directory for the plots.
    """
    file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)

    for model, runs in resolutions.items():
        bins = np.linspace(0.0, 1.0, 11)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        all_percentages = []

        # Plot individual resolutions
        for run in runs:
            # Get all predictions and calculate the percentage of predictions in each bin
            all_preds = np.concatenate([run[0], run[1]])
            total_predictions = len(all_preds)

            counts, _ = np.histogram(all_preds, bins=bins)

            if total_predictions > 0:
                percentage = (counts / total_predictions) * 100
            else:
                percentage = np.zeros_like(counts, dtype=float)

            all_percentages.append(percentage)

            plt.plot(
                bin_centers,
                percentage,
                color="lightgray",
                linewidth=1.5,
                alpha=0.6,
            )

        # Plot the mean resolution
        if all_percentages:
            mean_pct = np.mean(all_percentages, axis=0)

            plt.plot(
                bin_centers,
                mean_pct,
                color="#4f4e9b",
                linewidth=3,
                marker="o",
                label="Average Resolution",
                zorder=5,
            )

        plt.title(f"Resolution Plot ({model})")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Percentage of Total Predictions (%)")
        plt.xticks(bins)
        plt.legend(loc="upper center")
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(file_path / f"{model}_resolution.png", dpi=300)
        plt.show()
        plt.close()
