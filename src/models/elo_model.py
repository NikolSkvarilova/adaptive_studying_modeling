from collections import defaultdict

import numpy as np
import pandas as pd

from src.models.base_model import OptimizableModel


class EloModel(OptimizableModel):
    """ELO model for predicting student performance. It maintains separate ELO ratings for students and items, 
    updating them based on interactions."""

    def __init__(self, default_elo: float = 1000, k_factor: int = 16) -> None:
        """
        Args:
            default_elo (float, optional): default ELO rating. Defaults to 1000.
            k_factor (int, optional): K-factor for ELO updates. Defaults to 16. Optimizable via Optuna.
        """
        super().__init__()

        self.default_elo = default_elo
        self.k_factor = k_factor
        self.reset()

    def reset(self) -> None:
        self.item_dictionary = defaultdict(lambda: self.default_elo)
        self.student_dictionary = defaultdict(lambda: self.default_elo)

    @property
    def optimized_parameters(self):
        return {"k_factor": lambda trial: trial.suggest_float("k_factor", 5, 60)}

    def _process_interaction(self, student: int, item: int, correct: int) -> float:
        """Process a single interaction between a student and an item. Update the student's and item's ELO rating.

        Args:
            student (int): student ID
            item (int): item ID
            correct (int): correctness of the response (1 for correct, 0 for incorrect)

        Returns:
            float: probability of the student answering correctly
        """
        student_elo = self.student_dictionary[student]
        item_elo = self.item_dictionary[item]

        # Compute the probability of student answering correctly
        probability = 1 / (1 + 10 ** ((item_elo - student_elo) / 400))
        self.student_dictionary[student] += self.k_factor * (correct - probability)
        self.item_dictionary[item] += self.k_factor * (probability - correct)

        return probability

    def fit(self, train: pd.DataFrame) -> None:
        students = train["user"]
        items = train["item"]
        corrects = train["correct"]

        for student, item, correct in zip(students, items, corrects):
            self._process_interaction(student, item, correct)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        students = test["user"]
        items = test["item"]
        corrects = test["correct"]
        return np.array(
            [
                self._process_interaction(student, item, correct)
                for student, item, correct in zip(students, items, corrects)
            ]
        )
