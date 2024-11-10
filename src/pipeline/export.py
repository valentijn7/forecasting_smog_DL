# src/pipeline/export.py

# Functions that export processed data to the data directories,
# to be used for modelling (or in the pipeline itself) later

import pandas as pd


def export_minmax(df: pd.DataFrame, filename: str) -> None:
    """
    Exports minmax normalised dataframe to csv file

    :param df: dataframe
    """
    df.to_csv(f"../data/data_combined/{filename}.csv", 
               index = True, sep = ';', decimal = '.', encoding = 'utf-8')