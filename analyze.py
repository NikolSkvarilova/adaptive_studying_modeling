import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.misc.data_split import split_temporal_global
from src.misc.dataset import load_dataset
from src.models.elo_model import EloModel
from src.models.ensemble_model import EnsembleModel
from src.models.item_average import ItemAverageModel
from src.models.user_exponential_moving_avg import UserExpMovingAvg


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):  # np.float64, np.int32
            return obj.item()
        return super().default(obj)


if __name__ == "__main__":
    path = Path("./data")
    models = [
        ItemAverageModel,
        UserExpMovingAvg,
        EloModel,
        lambda: EnsembleModel([ItemAverageModel(), UserExpMovingAvg()]),
        lambda: EnsembleModel([ItemAverageModel(), UserExpMovingAvg(), EloModel()]),
    ]

    result_dir = Path("./results/")
    shutil.rmtree(result_dir)
    result_dir.mkdir()

    optimization_trials = 25
    results = []
    calibrations = defaultdict(list)
    resolutions = defaultdict(list)
    csv_files = list(path.rglob("*.csv"))

    # Iterate through all datasets and models, fit the models, and evaluate their performance
    for file_path in tqdm(csv_files, desc="Processing Datasets"):
        category = file_path.parent.parent.name
        dataset = load_dataset(file_path)
        train, validation, test = split_temporal_global(
            dataset, train_portion=0.6, validation_portion=0.2
        )

        # Fit each model, optimize parameters if possible, and evaluate performance
        for model_class in models:
            model = model_class()
            if model.optimizable:
                model.optimize_parameters(train, validation, optimization_trials)

            model.fit(pd.concat([train, validation], ignore_index=True))
            predicted = model.predict(test)
            actual = test["correct"]
            eval = model.eval(predicted, actual)

            eval["model"] = str(model)
            eval["dataset"] = file_path.name
            eval["category"] = category
            results.append(eval)

            resolutions[str(model)].append(
                model.resolution_plot(
                    predicted,
                    actual,
                    path=result_dir
                    / f"resolution_plot/{category}/{file_path.stem}_{model}.png",
                )
            )

            calibrations[str(model)].append(
                model.calibration_plot(
                    predicted,
                    actual,
                    path=result_dir
                    / f"calibration_plot/{category}/{file_path.stem}_{model}.png",
                )
            )

    # Save results, calibrations, and resolutions to CSV and JSON files
    results = pd.DataFrame(results)
    results.to_csv(result_dir / "results.csv", index=False)

    json_filepath = Path(result_dir / "json_files/")
    json_filepath.mkdir(parents=True, exist_ok=True)
    with open(json_filepath / "calibrations.json", "w") as f:
        json.dump(calibrations, f, cls=NumpyEncoder)
    with open(json_filepath / "resolutions.json", "w") as f:
        json.dump(resolutions, f, cls=NumpyEncoder)
