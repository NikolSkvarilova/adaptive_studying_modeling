from __future__ import annotations

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Read csv dataset. Drop responseTime, rank columns.

    Args:
        path (Path): path to dataset.

    Returns:
        pd.DataFrame: dataset
    """
    dataset = pd.read_csv(path)
    dataset = dataset.drop(columns=["responseTime", "rank"])
    return dataset
