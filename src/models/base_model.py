import copy
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score


class Model(ABC):
    """Base model class. Implements plotting and evaluation methods."""

    def __init__(self) -> None:
        self.optimizable = False

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None:
        """Fit the model to the training data.

        Args:
            train (pd.DataFrame): sorted training data.
        """

    @abstractmethod
    def predict(self, test: pd.DataFrame) -> np.ndarray:
        """Use the model to make predictions over the testing data.

        Args:
            test (pd.DataFrame): sorted testing data.

        Returns:
            np.ndarray: array of predictions.
        """

    def __repr__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def eval(predicted: np.ndarray, actual: np.ndarray) -> dict:
        """Computes evaluation statistics.

        Args:
            predicted (np.ndarray): array of predictions.
            actual (np.ndarray): array of actual values.

        Returns:
            dict: dictionary of evaluation metrics.
        """
        return {
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "auc": roc_auc_score(actual, predicted),
            "brier": brier_score_loss(actual, predicted),
        }

    def resolution_plot(
        self, predicted: np.ndarray, actual: np.ndarray, path: Path = None
    ) -> list:
        """Create the resolution plot, showing the distribution of predictions.

        Args:
            predicted (np.ndarray): array of predictions.
            actual (np.ndarray): array of actual values.
            path (Path, optional): path to save the plot. Defaults to None.

        Returns:
            list: list containing arrays of incorrect and correct predictions. Used for overlaying multiple resolution plots.
        """

        bins = np.linspace(0.0, 1.0, 11)

        pred_correct = predicted[actual == 1]
        pred_incorrect = predicted[actual == 0]

        total_predictions = len(predicted)

        weights_incorrect = np.ones_like(pred_incorrect) / total_predictions * 100
        weight_correct = np.ones_like(pred_correct) / total_predictions * 100

        plt.figure(figsize=(8, 5))
        plt.hist(
            [pred_incorrect, pred_correct],
            bins=bins,
            stacked=True,
            weights=[weights_incorrect, weight_correct],
            edgecolor="black",
            color=["#ff9999", "#66b3ff"],
            label=["Actual: Incorrect", "Actual: Correct"],
        )

        plt.title(f"Resolution Plot ({self})")
        plt.xlabel("Predicted Probability")
        plt.ylabel("% of predictions")
        plt.xticks(bins)
        plt.legend(loc="upper right")
        plt.grid(axis="y", alpha=0.3)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path)
        plt.close()

        return [pred_incorrect, pred_correct]

    def calibration_plot(
        self, predicted: np.ndarray, actual: np.ndarray, path: Path = None
    ) -> list:
        """Create the calibration plot.

        Args:
            predicted (np.ndarray): array of predictions.
            actual (np.ndarray): array of actual values.
            path (Path, optional): path to save the plot. Defaults to None.

        Returns:
            list: list containing arrays of predicted and actual means. Used for overlaying multiple calibration plots.
        """
        bins = np.linspace(0.0, 1.0, 11)

        bin_assignments = np.digitize(predicted, bins) - 1

        bin_predicted_means = []
        bin_actual_means = []

        for i in range(10):
            in_this_bin = bin_assignments == i

            if np.any(in_this_bin):
                bin_predicted_means.append(np.mean(predicted[in_this_bin]))
                bin_actual_means.append(np.mean(actual[in_this_bin]))

        plt.figure(figsize=(8, 5))
        plt.plot(
            bin_predicted_means,
            bin_actual_means,
            marker="o",
            color="#2ca02c",
            linewidth=2,
            label="Model",
        )
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration"
        )

        plt.title(f"Calibration Plot ({self})")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Actual Fraction of Correct Answers")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path)
        plt.close()

        return [bin_predicted_means, bin_actual_means]


class OptimizableModel(Model):
    """Base model class for models that can be optimized using Optuna."""
    def __init__(self):
        super().__init__()
        self.optimizable = True

    @abstractmethod
    def reset(self) -> None:
        """Reset the model's inner state."""
        pass

    @property
    @abstractmethod
    def optimized_parameters(self):
        """Returns a dictionary with Optuna-ready parameters. The keys are the parameter names, 
        and the values are functions that take an Optuna trial and return a suggested value."""
        pass

    def optimize_parameters(
        self, train: pd.DataFrame, validation: pd.DataFrame, n_trials: int
    ) -> None:
        """Optimize the parameters using Optuna.

        Args:
            train (pd.DataFrame): training data
            validation (pd.DataFrame): validation data
            n_trials (int): number of optimization trials
        """
        def objective(trial: optuna.Trial) -> float:

            trial_model = copy.deepcopy(self)
            trial_model.reset()

            for param_name, suggest_func in trial_model.optimized_parameters.items():
                suggested_value = suggest_func(trial)
                setattr(trial_model, param_name, suggested_value)

            trial_model.fit(train)
            predicted = trial_model.predict(validation)
            actual = validation["correct"]

            return trial_model.eval(predicted, actual)["rmse"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        self.reset()

        # Set the best parameters found
        for key, value in study.best_params.items():
            setattr(self, key, value)
