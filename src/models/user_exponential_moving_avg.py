from collections import defaultdict

import numpy as np
import pandas as pd

from src.models.base_model import OptimizableModel


class UserExpMovingAvg(OptimizableModel):
    """User exponential moving average model for predicting student performance. 
    It maintains a dictionary of all students and their exponentially weighted 
    average correctness, which is used for predictions. Unknown students are assigned 
    a default probability of 0.5."""

    def __init__(self, alpha: float = 0.15) -> None:
        """
        Args:
            alpha (float, optional): The smoothing factor. Defaults to 0.15. Optimizable via Optuna.
        """
        super().__init__()

        self.alpha = alpha
        self.reset()

    def reset(self):
        self.dictionary = defaultdict(lambda: 0.5)

    @property
    def optimized_parameters(self):
        return {"alpha": lambda trial: trial.suggest_float("alpha", 0.05, 0.95)}

    def _process_interaction(self, student: int, correct: int) -> float:
        """Process a single interaction of a student. Update the student's  exponential moving average.

        Args:
            student (int): student ID
            correct (int): correctness of the response (1 for correct, 0 for incorrect)

        Returns:
            float: probability of the student answering correctly
        """
        current = self.dictionary[student]
        self.dictionary[student] = (self.alpha * correct) + (1 - self.alpha) * current
        return current

    def fit(self, train: pd.DataFrame) -> None:
        students = train["user"]
        corrects = train["correct"]

        for student, correct in zip(students, corrects):
            self._process_interaction(student, correct)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        students = test["user"]
        corrects = test["correct"]

        return np.array(
            [
                self._process_interaction(student, correct)
                for student, correct in zip(students, corrects)
            ]
        )
