# src/modelling/extract.py

# Functions to extract the data from the data/ folder

from typing import List
import pandas as pd


def import_csv(filename: str) -> pd.DataFrame:
    """
    Imports a file from the data/data_combined folder
    
    :param file_name: name of the file to import
    """
    return pd.read_csv(f'../data/data_combined/{filename}',
                       index_col = 'DateTime',
                       sep = ';',
                       decimal = '.')


def get_dataframes(what: str, UY: str) -> List[pd.DataFrame]:
    """
    Convenience function what based on what (= 'train', 'val',
    'test') and UY (= 'input' or 'output') returns the associated
    list of dataframes from the data/data_combined folder

    :param what: 'train', 'val' (= validation), 'test'
    :param UY: 'u' (= input), 'y' (= output)
    """
    if UY == 'u':
        if what == 'train':
            return [import_csv('train_2017_combined_u.csv'),
                    import_csv('train_2018_combined_u.csv'),
                    import_csv('train_2020_combined_u.csv'),
                    import_csv('train_2021_combined_u.csv'),
                    import_csv('train_2022_combined_u.csv')]
        if what == 'val':
            return [import_csv('val_2021_combined_u.csv'),
                    import_csv('val_2022_combined_u.csv'),
                    import_csv('val_2023_combined_u.csv')]
        if what == 'test':
            return [import_csv('test_2021_combined_u.csv'),
                    import_csv('test_2022_combined_u.csv'),
                    import_csv('test_2023_combined_u.csv')]
    if UY == 'y':
        if what == 'train':
            return [import_csv('train_2017_combined_y.csv'),
                    import_csv('train_2018_combined_y.csv'),
                    import_csv('train_2020_combined_y.csv'),
                    import_csv('train_2021_combined_y.csv'),
                    import_csv('train_2022_combined_y.csv')]
        if what == 'val':
            return [import_csv('val_2021_combined_y.csv'),
                    import_csv('val_2022_combined_y.csv'),
                    import_csv('val_2023_combined_y.csv')]
        if what == 'test':
            return [import_csv('test_2021_combined_y.csv'),
                    import_csv('test_2022_combined_y.csv'),
                    import_csv('test_2023_combined_y.csv')]
    raise ValueError(f"Invalid 'what' ({what}) or 'UY' ({UY}) parameter")