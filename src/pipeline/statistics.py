# src/pipeline/statistics.py

# This script contains functions which for simple data statistics.
# A lot of these get used in plots.py

from typing import List
import pandas as pd


def get_min_sensor_value(df: pd.DataFrame, sensor: str) -> float:
    """
    Get the minimum value of a sensor in the dataframe

    :param df: dataframe
    :param sensor: sensor name
    :return: minimum value of the sensor
    """
    return df[sensor].min()


def get_mean_sensor_value(df: pd.DataFrame, sensor: str) -> float:
    """ 
    Get the mean value of a sensor in the dataframe
    
    :param df: dataframe
    :param sensor: sensor name
    :return: mean value of the sensor
    """
    return df[sensor].mean()


def get_max_sensor_value(df: pd.DataFrame, sensor: str) -> float:
    """
    Get the maximum value of a sensor in the dataframe

    :param df: dataframe
    :param sensor: sensor name
    :return: maximum value of the sensor
    """
    return df[sensor].max()


def get_min_per_day(df: pd.DataFrame, sensor: str) -> pd.Series:
    """ 
    Get the minimum value of a sensor per day
    
    :param df: dataframe
    :param sensor: sensor name
    :return: minimum value of the sensor per day
    """
    return pd.Series(data = df[sensor].resample('D', origin = 'start').min())


def get_mean_per_day(df: pd.DataFrame, sensor: str) -> pd.Series:
    """ 
    Get the mean value of a sensor per day
    
    :param df: dataframe
    :param sensor: sensor name
    :return: mean value of the sensor per day
    """
    return pd.Series(data = df[sensor].resample('D', origin = 'start').mean())


def get_max_per_day(df: pd.DataFrame, sensor: str) -> pd.Series:
    """
    Get the maximum value of a sensor per day

    :param df: dataframe
    :param sensor: sensor name
    :return: maximum value of the sensor per day
    """
    return pd.Series(data = df[sensor].resample('D', origin = 'start').max())


def get_min_per_month(df: pd.DataFrame, sensor: str) -> pd.Series:
    """
    Get the minimum value of a sensor per month:      
    resample by month, take min(), shift to 15th of month

    :param df: dataframe
    :param sensor: sensor name
    :return: minimum value of the sensor per month
    """
    return pd.Series(data = \
                     df[sensor].resample('MS', convention = 'start').min().shift(14, 'D'))
                     

def get_mean_per_month(df: pd.DataFrame, sensor: str) -> pd.Series:
    """
    Get the mean value of a sensor per month:
    resample by month, take mean(), shift to 15th of month

    :param df: dataframe
    :param sensor: sensor name
    :return: mean value of the sensor per month
    """
    return pd.Series(data = \
                     df[sensor].resample('MS', convention = 'start').mean().shift(14, 'D'))


def get_max_per_month(df: pd.DataFrame, sensor: str) -> pd.Series:
    """
    Get the maximum value of a sensor per month:
    resample by month, take max(), shift to 15th of month

    :param df: dataframe
    :param sensor: sensor name
    :return: maximum value of the sensor per month
    """
    return pd.Series(data = \
                     df[sensor].resample('MS', convention = 'start').max().shift(14, 'D'))


def get_col_measurement_count(df: pd.DataFrame, col: str) -> int:
    """
    Returns number of measurements in a column (excluding NaNs)
    
    :param df: dataframe
    :param col: column name
    :return: number of measurements in the column
    """
    return df[col].count()


def print_index_sampling_info(df: pd.DataFrame) -> None:
    """
    Prints various sampling metrics of the index:
    - sample time distribution;
    - most frequent sample time; and
    - mean sample time.
    
    :param df: dataframe
    """
    print(f'Sample time distribution  =\n{df.index.to_series().diff().value_counts()}')
    print(f'Most frequent sample time = {df.index.to_series().diff().median()}')
    print(f'Mean sample time          = {df.index.to_series().diff().mean()}')


def print_sensor_metrics_min_mean_max_entries(df: pd.DataFrame, sensor: str, meta: dict) -> None:
    """
    Prints the min, mean, max, and number of entries of a sensor

    :param df: dataframe
    :param sensor: sensor name
    :param meta: dictionary with metadata
    """
    if not sensor in df.columns:
        return print(f"{meta['comp']} measurements for sensor {sensor} are not avaiable\n")

    print(f"[min, mean, max] for sensor {sensor} measuring {meta['comp']} {meta['unit']}:")
    print(f"[{get_min_sensor_value(df, sensor):.4f}, {get_mean_sensor_value(df, sensor):.4f},", end = ' ')
    print(f"{get_max_sensor_value(df, sensor):.4f}] with n = {get_col_measurement_count(df, sensor)}")
    print()


def print_aggegrated_sensor_metrics(dfs: List[pd.DataFrame], sensor: str, meta: dict) -> None:
    """
    Takes in a list of dataframes spanning multiple years and prints
    the aggregated min, mean, max, and number of entries of a sensor
    within these dataframes, similar to print_sensor_metrics_min_mean_max_entries()

    :param dfs: list of dataframes
    :param sensor: sensor name
    :param meta: dictionary with metadata
    """
    # First check if type and sensor column, then concatenate and print
    if not all([sensor in df.columns for df in dfs]):
        return print(f"print_aggregated_sensor_metrics(): Sensor {sensor} is not avaiable\n")
    df = pd.concat(dfs)

    print(f"[min, mean, max] for sensor {sensor} measuring {meta['comp']} {meta['unit']}")
    print(f"aggregated over multiple years:")
    print(f"[{get_min_sensor_value(df, sensor):.4f}, {get_mean_sensor_value(df, sensor):.4f},", end = ' ')
    print(f"{get_max_sensor_value(df, sensor):.4f}] with n = {get_col_measurement_count(df, sensor)}")
    print()


def get_daily_sensor_metrics(df: pd.DataFrame, sensor: str) -> float:
    """ 
    Helper function to get the min, mean, and max values of a sensor per day
    
    :param df: dataframe
    :param sensor: sensor name
    :return: tuple of min, mean, and max values of the sensor per day
    """
    return get_min_per_day(df, sensor), get_mean_per_day(df, sensor), get_max_per_day(df, sensor)


def get_monthly_sensor_metrics(df: pd.DataFrame, sensor: str) -> float:
    """
    Helper function to get the min, mean, and max values of a sensor per month

    :param df: dataframe
    :param sensor: sensor name
    :return: tuple of min, mean, and max values of the sensor per month
    """
    return get_min_per_month(df, sensor), get_mean_per_month(df, sensor), get_max_per_month(df, sensor)