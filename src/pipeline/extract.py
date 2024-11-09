# src/pipeline/extract.py

# This file contains code which functions as the first step in the pipeline;
# its functions extract data from the source and return it as a DataFrame

import os
import sys
import pandas as pd


def set_working_directory(path: str, device: str) -> None:
    """
    Plainly sets the current working directory to the specified path.
    Also checks what the current device is, because directory lay-out
    might differ between machines (e.g. Linux vs Windows).

    :param path: directory path
    :param device: device name
    """
    if device == 'tinus':
        os.chdir(path)
    elif device == 'habrok':
        os.chdir(path)
    ### Add more devices here if needed
    else:
        sys.exit("Device not recognised. Please enter a valid device name.")



def read_contaminant_csv_from_data_raw(
        component: str, year: str, path: str, device: str, rows_to_skip: int = 9
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
    set_working_directory(path, device)
                                                      
    return pd.read_csv(f"../data/data_raw/{year}_{component}.csv", 
                       sep = ';',
                       encoding = 'ISO-8859-15',
                       skiprows = rows_to_skip)


def read_meteo_csv_from_data_raw(year: str, path: str, device: str) -> pd.DataFrame:
    """
    Reads the meteorological data from the raw data folder

    :param year: the year of the data to be read
    :param path: the path to the data folder
    :param device: the device name
    :return: the meteorological data as a pandas DataFrame
    """
    set_working_directory(path, device)

    return pd.read_csv(f"../data/data_raw/{year}_meteo_Utrecht.csv",
                       sep = ';',
                       encoding = 'UTF-8',
                       index_col = 0) 
                                                          
                                                            
def read_four_contaminants(
        year: str, contaminants: str, path: str, device: str
    ) -> pd.DataFrame:
    """
    Helper function for downloading four contaminant dataframes at once
    
    :param year: the year of the data to be read
    :param contaminants: a list of the four contaminants to be read
    :param path: the path to the data folder
    :param device: the device name
    :return: four contaminant dataframes
    """
    df1 = read_contaminant_csv_from_data_raw(contaminants[0], year, path, device)
    df2 = read_contaminant_csv_from_data_raw(contaminants[1], year, path, device)
    df3 = read_contaminant_csv_from_data_raw(contaminants[2], year, path, device)
    df4 = read_contaminant_csv_from_data_raw(contaminants[3], year, path, device)
    return df1, df2, df3, df4


def read_two_meteo_years(yr1: str, yr2: str, path: str, device: str) -> pd.DataFrame:
    """
    Helper function for downloading two meteorological dataframes at once
    
    :param yr1: the first year of the data to be read
    :param yr2: the second year of the data to be read
    :param path: the path to the data folder
    :param device: the device name
    :return: two meteorological dataframes
    """
    df1 = read_meteo_csv_from_data_raw(yr1, path, device)
    df2 = read_meteo_csv_from_data_raw(yr2, path, device)
    return df1, df2