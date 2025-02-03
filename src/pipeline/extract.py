# src/pipeline/extract.py

# This file contains code which functions as the first step in the pipeline;
# its functions extract data from the source and return it as a DataFrame

import os
from pathlib import Path
import sys
import pandas as pd


def read_contaminant_csv_from_data_raw(
        component: str,
        year: str,
        rows_to_skip: int = 9
    ) -> pd.DataFrame:
    """
    Reads the contaminant data from the raw data folder. The data is in CSV format.
    The data contains 9 rows of metedata, which are skipped. These can be loaded in
    by adjusting the default value of the 'rows_to_skip' parameter, but will cause
    errors further down the pipeline if not handled correctly. Easier would be to
    inspect the .csv files manuallt if any metadata is needed. Or, read the metadata
    separately. The encoding is set to ISO-8859-15, as specified in the pdf:
    
    https://data.rivm.nl/data/luchtmeetnet/readme.pdf

    :param component: the contaminant to be read
    :param year: the year of the data to be read
    :param path: the path to the data folder
    :param device: the device name
    :param rows_to_skip: the number of rows to skip in the CSV file
    :return: the contaminant data as a pandas DataFrame
    """
    os.chdir(Path.cwd())
                                                      
    return pd.read_csv(f"../data/data_raw/{year}_{component}.csv", 
                       sep = ';',
                       encoding = 'ISO-8859-15',
                       skiprows = rows_to_skip)


def read_meteo_csv_from_data_raw(
        year: str
    ) -> pd.DataFrame:
    """
    Reads the meteorological data from the raw data folder

    :param year: the year of the data to be read
    :param path: the path to the data folder
    :param device: the device name
    :return: the meteorological data as a pandas DataFrame
    """
    os.chdir(Path.cwd())

    return pd.read_csv(f"../data/data_raw/{year}_meteo_Utrecht.csv",
                       sep = ';',
                       encoding = 'UTF-8',
                       index_col = 0) 
                                                          
                                                            
def read_four_contaminants(
        year: str,
        contaminants: str
    ) -> pd.DataFrame:
    """
    Helper function for downloading four contaminant dataframes at once
    
    :param year: the year of the data to be read
    :param contaminants: a list of the four contaminants to be read
    :param path: the path to the data folder
    :param device: the device name
    :return: four contaminant dataframes
    """
    df1 = read_contaminant_csv_from_data_raw(contaminants[0], year)
    df2 = read_contaminant_csv_from_data_raw(contaminants[1], year)
    df3 = read_contaminant_csv_from_data_raw(contaminants[2], year)
    df4 = read_contaminant_csv_from_data_raw(contaminants[3], year)
    return df1, df2, df3, df4


def read_two_meteo_years(
        yr1: str,
        yr2: str
    ) -> pd.DataFrame:
    """
    Helper function for downloading two meteorological dataframes at once
    
    :param yr1: the first year of the data to be read
    :param yr2: the second year of the data to be read
    :param path: the path to the data folder
    :param device: the device name
    :return: two meteorological dataframes
    """
    return read_meteo_csv_from_data_raw(yr1), read_meteo_csv_from_data_raw(yr2)