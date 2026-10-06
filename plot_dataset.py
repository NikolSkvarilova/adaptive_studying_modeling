from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.misc.dataset import load_dataset

if __name__ == "__main__":
    path = Path("./data")
    result_dir = Path("./results/")
    result_dir.mkdir(exist_ok=True, parents=True)
    csv_files = list(path.rglob("*.csv"))

    corrects = []

    for file_path in csv_files:
        dataset = load_dataset(file_path)

        corrects.append(dataset["correct"].to_list())

    proportions = [sum(c) / len(c) for c in corrects]

    plt.hist(proportions, bins=np.arange(0, 1, 0.1), color="blue", edgecolor="black")
    plt.xlabel("Proportion of Correct Answers")
    plt.title("Distribution of Proportions Across Datasets")
    plt.ylabel("# Datasets")

    plt.savefig(result_dir / "proportion_distribution.png", dpi=300)
