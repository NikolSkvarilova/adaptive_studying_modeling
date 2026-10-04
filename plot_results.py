import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.misc.plot import plot_all_calibrations, plot_all_resolutions

if __name__ == "__main__":
    df = pd.read_csv("./results/results.csv")
    sns.boxplot(data=df, x="model", y="auc", color="lightgray", showfliers=False)
    sns.stripplot(data=df, x="model", y="auc", jitter=0.2, alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.title("AUC achieved over all datasets")

    plt.savefig("./results/auc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------

    sns.boxplot(data=df, x="model", y="rmse", color="lightgray", showfliers=False)
    sns.stripplot(data=df, x="model", y="rmse", jitter=0.2, alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.title("RMSE achieved over all datasets")

    plt.savefig("./results/rmse_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------

    # Plotting global calibrations for every model
    with open("./results/json_files/calibrations.json", "r") as f:
        calibrations = json.load(f)

    with open("./results/json_files/resolutions.json", "r") as f:
        resolutions = json.load(f)

    plot_all_calibrations(calibrations, "./results/models/")
    plot_all_resolutions(resolutions, "./results/models/")
