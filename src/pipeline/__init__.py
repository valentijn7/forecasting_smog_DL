# src/pipeline/__init__.py

# __init__py for the imperatively programmed data pipeline used
# in the paper: https://icml.cc/virtual/2024/36782

__version__ = '0.0.0' # MAJOR.MINOR.PATCH versioning
__author__ = 'valentijn7' # GitHub username

print("\nRunning __init__.py for data pipeline")

from .extract import read_contaminant_csv_from_data_raw
from .extract import read_meteo_csv_from_data_raw
from .extract import read_four_contaminants
from .tidy import get_metadata
from .tidy import tidy_raw_contaminant_data
from .tidy import tidy_raw_meteo_data
from .statistics import print_sensor_metrics_min_mean_max_entries
from .statistics import get_daily_sensor_metrics
from .statistics import get_monthly_sensor_metrics

print("Pipeline initialized\n")