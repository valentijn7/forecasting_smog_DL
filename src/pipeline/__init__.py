# src/pipeline/__init__.py

# __init__py for the imperatively programmed data pipeline package

__version__ = '0.0.0' # MAJOR.MINOR.PATCH versioning
__author__ = 'valentijn7' # GitHub username

print("\nRunning __init__.py for data pipeline")

from .extract import read_contaminant_csv_from_data_raw
from .extract import read_meteo_csv_from_data_raw
from .extract import read_four_contaminants
from .tidy import get_metadata
from .tidy import tidy_raw_contaminant_data
from .tidy import tidy_raw_meteo_data
from .transform import subset_sensors
from .split import perform_data_split
from .split import perform_data_split_without_train
from .split import print_split_ratios
from .statistics import print_sensor_metrics_min_mean_max_entries
from .statistics import print_aggegrated_sensor_metrics
from .statistics import get_daily_sensor_metrics
from .statistics import get_monthly_sensor_metrics
from .normalise import get_df_minimum
from .normalise import get_df_maximum
from .normalise import calc_combined_min_max_params
from .normalise import normalise_linear
from .normalise import normalise_linear_inv
from .normalise import print_pollutant_extremes
from .export import export_minmax
from .transform import make_manual_dict_of_dfs
from .transform import sort_dict_of_dfs
from .transform import print_dict_of_dfs
from .transform import print_dfs_sorted
from .transform import concat_frames_horizontally
from .transform import delete_timezone_from_index
from .tests import assert_equal_shape
from .tests import assert_equal_index
from .tests import assert_no_NaNs
from .tests import assert_range
from .plots import plot_sensor
from .plots import plot_sensor_meta
from .plots import plot_min_mean_max
from .plots import plot_day_vs_month
from .plots import plot_corr_matrix
from .plots import plot_distributions_KDE
from .plots import plot_multiple_distributions
from .plots import plot_tails
from .pipeline import execute_pipeline

print("Pipeline initialized\n")