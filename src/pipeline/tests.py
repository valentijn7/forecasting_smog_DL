# src/pipeline/tests.py

# Contains some test(s) that check whether some data transformations
# have been done correctly. These tests are definitely not exhaustive,
# but should give a good indication of whether the pipeline is working

from typing import List
import pandas as pd


def assert_equal_shape(
        dfs: List[pd.DataFrame], check_rows: bool, check_cols: bool, test_point: str) -> None:
    """
    Check if the dataframes in the list have the same shape

    :param dfs: list of dataframes
    :param check_rows: whether to check the number of rows
    :param check_cols: whether to check the number of columns
    """
    if check_rows:
        assert all([df.shape[0] == dfs[0].shape[0] for df in dfs]), \
            f"@ {test_point}: test failed, unequal row count between dataframes"
    if check_cols:
        assert all([df.shape[1] == dfs[0].shape[1] for df in dfs]), \
            f"@ {test_point}: test failed, unequal column count between dataframes"
        

def assert_equal_index(dfs: List[pd.DataFrame], test_point: str) -> None:
    """
    Check if the dataframes in the list have the same index

    :param dfs: list of dataframes
    """
    assert all([df.index.equals(dfs[0].index) for df in dfs]), \
        f"@ {test_point}: test failed, unequal index between dataframes"
        

def assert_no_NaNs(dfs: List[pd.DataFrame], test_point: str) -> None:
    """
    Check if there are any NaNs in the dataframes in the list

    :param dfs: list of dataframes
    """
    assert all([df.isnull().sum().sum() == 0 for df in dfs]), \
        f"@ {test_point}: test failed, NaNs found in dataframes"
    
def assert_range(
        dfs: List[pd.DataFrame], min_val: float,
        max_val: float, test_point: str
    ) -> None:
    """
    Check if all values in the dataframes in the list are within the specified range

    :param dfs: list of dataframes
    :param min_val: minimum value
    :param max_val: maximum value
    """
    assert all([df.values.min() >= min_val for df in dfs]), \
        f"@ {test_point}: test failed, values below minimum"
    assert all([df.values.max() <= max_val for df in dfs]), \
        f"@ {test_point}: test failed, values above maximum"