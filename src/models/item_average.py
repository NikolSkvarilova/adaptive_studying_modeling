from collections import defaultdict

import numpy as np
import pandas as pd

from src.models.base_model import Model


class ItemAverageModel(Model):
    """Item average model for predicting student performance. It maintains a dictionary of all items and their average correctness, 
    which is used for predictions. Unknown items are assigned a default probability of 0.5."""
    def __init__(self) -> None:
        super().__init__()

        # dictionary with items and their probabilities of being correct
        # unknown items get probability 0.5
        self.dictionary = defaultdict(lambda: 0.5)

    def fit(self, train: pd.DataFrame) -> None:
        # mean is computed as correct / total attempts
        item_averages = train.groupby("item")["correct"].mean().to_dict()
        self.dictionary.update(item_averages)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.array([self.dictionary[item] for item in test["item"]])
