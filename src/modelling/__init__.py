# src/modelling/__init__.py

# __init__py for the modelling package

__version__ = '0.0.0' # MAJOR.MINOR.PATCH versioning
__author__ = 'valentijn7' # GitHub username

print("\nRunning __init__.py for data pipeline")

from .extract import import_csv
from .extract import get_dataframes
from .EarlyStopper import EarlyStopper
from .TimeSeriesDataset import TimeSeriesDataset
from .PrintManager import PrintManager

print("Modelling package initialized\n")