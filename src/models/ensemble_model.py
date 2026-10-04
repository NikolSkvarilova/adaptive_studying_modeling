from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from src.models.base_model import OptimizableModel


class EnsembleModel(OptimizableModel):
    """Ensemble model that combines multiple models. It allows for weighted averaging of predictions 
    from different models, and can optimize the weights using Optuna."""

    def __init__(self, models: list[Model]) -> None:
        """
        Args:
            models (list[Model]): list of initialized models to be combined in the ensemble.
        """
        
        super().__init__()

        self.models = models
        self.optimizable = bool(sum([model.optimizable for model in self.models]))
        for i in range(len(self.models)):
            setattr(self, f"weight_{i}", 1.0)

    def reset(self) -> None:
        for model in self.models:
            model.reset()

    @property
    def optimized_parameters(self):
        return {
            f"weight_{i}": lambda trial, i=i: trial.suggest_float(
                f"weight_{i}", 0.0, 1.0
            )
            for i in range(len(self.models))
        }

    def __repr__(self) -> str:
        return f"{'+'.join([f'{model}' for model in self.models])}"

    def fit(self, train: pd.DataFrame) -> None:
        for model in self.models:
            model.fit(train)

    def optimize_parameters(
        self, train: pd.DataFrame, validation: pd.DataFrame, n_trials: int
    ) -> None:
        """Ensemble models require custom optimization. First, the inner models are optimized (if possible).
        Then, the weights for the ensemble are optimized using Optuna.

        Args:
            train (pd.DataFrame): training data
            validation (pd.DataFrame): validation data
            n_trials (int): number of optimization trials
        """
        
        # Oprimize inner models first
        for model in self.models:
            if model.optimizable:
                model.optimize_parameters(train, validation, n_trials)

            model.fit(train)

        # Save the predictions from the inner models (validation set) for weight optimization
        val_preds_2d = np.array([model.predict(validation) for model in self.models])
        actual = validation["correct"]

        def objective(trial: optuna.Trial) -> float:
            trial_weights = [
                trial.suggest_float(f"weight_{i}", 0.0, 1.0)
                for i in range(len(self.models))
            ]

            if sum(trial_weights) == 0:
                trial_weights = [1.0 for _ in self.models]

            # Compute the weighted average of predictions
            predicted = np.average(val_preds_2d, axis=0, weights=trial_weights)

            return self.eval(predicted, actual)["rmse"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

        for key, value in study.best_params.items():
            setattr(self, key, value)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        weights = [getattr(self, f"weight_{i}") for i in range(len(self.models))]
        if sum(weights) == 0:
            weights = [1.0 for _ in self.models]
        predicted_2d = np.array([model.predict(test) for model in self.models])
        return np.average(predicted_2d, axis=0, weights=weights)
