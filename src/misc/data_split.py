import pandas as pd


def split_temporal_global(
    dataset: pd.DataFrame, *, train_portion: float, validation_portion: float = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort the dataset by timestamp and split. Test portion is what is left after train and validation is cut off.

    Args:
        dataset (pd.DataFrame): dataset
        train_portion (float): 0.0-1.0
        validation_portion (float, optional): 0.0-1.0. Defaults to None. If None, returned validation portion is None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train df, validation df, test df
    """
    sorted_dataset = dataset.sort_values(by="time")
    pivot = int(train_portion * len(sorted_dataset))

    train_df = sorted_dataset.iloc[:pivot].copy()
    validation_df = pd.DataFrame()
    if validation_portion:
        validation_size = int(len(sorted_dataset) * validation_portion)
        validation_df = sorted_dataset.iloc[pivot : pivot + validation_size]
        pivot += validation_size

    test_df = sorted_dataset.iloc[pivot:].copy()

    return train_df, validation_df, test_df
