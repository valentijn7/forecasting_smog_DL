# src/pipeline/plots.py

# A lot of plotting functions, including but not limited to:
# - plain timeseries plots of a certain sensor
# - timeseries plot with (rolling) min, mean, and max values
# - timeseries plot with daily and monthly mean values
# - ... TBC

from .statistics import get_daily_sensor_metrics
from .statistics import get_mean_per_day
from .statistics import get_mean_per_month

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    """ 
    Sets the style for the plots
    """
    sns.set_theme()
    sns.axes_style('darkgrid')
    sns.set_palette('dark') 
    sns.set_context('notebook')
    

def plot_sensor(df: pd.DataFrame, sensor: str, info: str = ''):
    """
    Plots all measurements for one sensor against time (i.e. no subsetting)

    :param df: DataFrame with the data
    :param sensor: the sensor to plot
    :param info: additional information to add to the title
    """
    if sensor not in df:
        return print(f"Sensor {sensor} is not avaiable\n")

    set_style()
    sns.lineplot(data = df, x = 'DateTime', y = sensor, color = '#800000')

    plt.title(f"Sensor {sensor} plotted against time - {info}")
    plt.xlabel("Time")
    plt.xticks(rotation = 18)
    plt.ylabel(f"Measurement value")
    plt.show()


def plot_sensor_meta(df: pd.DataFrame, sensor: str, meta: dict):
    """
    Plots all measurements for one sensor against time, with metadata

    :param df: DataFrame with the data
    :param sensor: the sensor to plot
    :param meta: metadata for the sensor
    """
    if sensor not in df:
        return print(f"{meta['comp']} measurements for sensor {sensor} are not avaiable\n")

    set_style()
    sns.lineplot(data = df, x = 'DateTime', y = sensor, color = '#800000')

    plt.title(f"Sensor {sensor} plotted against time")
    plt.xlabel("Time")
    plt.xticks(rotation = 18)
    plt.ylabel(f"{meta['comp']} value in {meta['unit']}")
    plt.show()

                                        
def plot_min_mean_max(df: pd.DataFrame, sensor: str, meta: dict):
    """
    Plots min, mean, max of a sensor against time

    :param df: DataFrame with the data
    :param sensor: the sensor to plot
    :param meta: metadata for the sensor
    """
    mins, means, maxs = get_daily_sensor_metrics(df, sensor)
    
    set_style()

    sns.lineplot(data = mins.to_frame(), x = mins.index, y = mins.values, 
                 label = 'min')#, color = '#FED116')
    sns.lineplot(data = means.to_frame(), x = means.index, y = means.values, 
                 label = 'mean')#, color = '#CD1127')
    sns.lineplot(data = maxs.to_frame(), x = maxs.index, y = maxs.values, 
                 label = 'max')#, color = '#013893')

    plt.title(f"Sensor {sensor}'s min, max, mean plotted against time")
    plt.xlabel("Time")
    plt.xticks(rotation = 18)
    plt.ylabel(f"{meta['comp']} value in {meta['unit']}")
    plt.show()


def plot_day_vs_month(df: pd.DataFrame, sensor: str, meta: dict):
    """
    Plots daily vs monthly average of a sensor against time

    :param df: DataFrame with the data
    :param sensor: the sensor to plot
    :param meta: metadata for the sensor
    """
    days = get_mean_per_day(df, sensor)
    mons = get_mean_per_month(df, sensor)

    set_style()

    sns.lineplot(data = days.to_frame(), x = days.index, 
                 y = days.values, label = 'day')
    plt.stem(mons.index, mons.values, basefmt = ' ', 
             linefmt = '--r', markerfmt = 'or', label = 'month')
    # for more customization visit:
    # https://stackoverflow.com/questions/38984959/how-can-i
    # -get-the-stemlines-color-to-match-the-marker-color-in-a-stem-plot

    plt.title(f"Sensor {sensor}'s daily and monthly average plotted against time")
    plt.xlabel("Time")
    plt.xticks(rotation = 18)
    plt.ylabel(f"{meta['comp']} value in {meta['unit']}")
    plt.legend()
    plt.show()
