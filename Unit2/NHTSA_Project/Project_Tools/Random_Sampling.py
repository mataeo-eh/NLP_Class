from __future__ import annotations

import pandas as pd

def get_available_rows(
    df: pd.DataFrame,
    processed_indices: set[int] | None = None,
) -> pd.DataFrame:
    """
    Return only rows whose dataframe index has not already been processed.
    """
    if not processed_indices:
        return df.copy()

    return df.loc[~df.index.isin(processed_indices)].copy()


def get_sequential_rows(
    df: pd.DataFrame,
    n: int,
    processed_indices: set[int] | None = None,
) -> pd.DataFrame:
    """
    Return the first n unprocessed rows in original dataframe order.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    available_rows = get_available_rows(df, processed_indices)
    return available_rows.head(n).copy()


def get_random_sample_rows(
    df: pd.DataFrame,
    n: int,
    processed_indices: set[int] | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Randomly sample n unprocessed rows without replacement.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if n > len(df):
        raise ValueError("n cannot be larger than the number of rows in the dataset")

    available_rows = get_available_rows(df, processed_indices)

    if n > len(available_rows):
        raise ValueError(
            "n cannot be larger than the number of unprocessed rows available for sampling"
        )

    return available_rows.sample(n=n, replace=False, random_state=random_state).copy()
