# src/pipeline/plots.py

# A lot of plotting functions, including but not limited to:
# - plain timeseries plots of a certain sensor
# - timeseries plot with (rolling) min, mean, and max values
# - timeseries plot with daily and monthly mean values
# - etc.

from .statistics import get_daily_sensor_metrics
from .statistics import get_mean_per_day
from .statistics import get_mean_per_month

from typing import List, Union
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    """ 
    Sets the style for the plots
    """
    mpl.rcParams['figure.figsize'] = (6, 2) # landscape plots

    sns.set_theme()
    sns.axes_style('darkgrid')
    sns.set_palette('dark') 
    sns.set_context('notebook')


def set_thesis_style() -> None:
    """
    Sets a more precise style suited for the thesis
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "axes.labelsize": 8,     # fontsize for x and y labels
        "xtick.labelsize": 8,    # fontsize of the tick labels
        "ytick.labelsize": 8,    # fontsize of the tick labels
        "legend.fontsize": 8,    # fontsize of the legen
    })
    
    sns.set_theme(style = 'ticks')
    sns.set_context("paper")
    

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


def plot_corr_matrix(df, threshold = 0, method = 'pearson'):
    """
    Uses a correlation matrix to show the correlation between the
    columns of a dataframe using the Peasron correlation coefficient.
    Other methods are 'kendall' and 'spearman', but 'pearson' is
    the default and also what was used in the thesis
    
    Source/inspiration: # https://seaborn.pydata.org/examples/many_pairwise_correlations.html

    :param df: dataframe
    :param threshold: threshold value for the correlation coefficient to show in the plot
    :param method: correlation coefficient calculation method
    """
    corr = df.corr(method)
    if threshold:
        corr = corr[corr.abs() > threshold]

    mask = np.triu(np.ones_like(corr, dtype = bool))

    f, ax = plt.subplots(figsize = (7, 5))
    # # cmap = sns.diverging_palette(230, 20, as_cmap = True)
    # cmap = sns.diverging_palette(0, 255, s = 100, sep = 1, as_cmap = True)

    # sns.heatmap(corr, mask = mask, cmap = cmap, center = 0,
    #             square = True, linewidths = .5, cbar_kws = {"shrink": .5});
    sns.heatmap(corr, mask = mask,
                vmin = -1, vmax = 1, center = 0,
                square = True, linewidth = .5, cbar_kws = {"shrink": .75})
    plt.tight_layout()
    plt.show()


def plot_distributions_KDE(data: Union[pd.Series, pd.DataFrame], title: str) -> None:
    """
    Plots the distribution of a sensor's measurements
    using KDE (Kernel Density Estimation)
    
    :param data: sensor's measurements
    :param title: title of the plot
    """
    set_style()
    
    if isinstance(data, pd.Series):      # distinguish between Series and DataFrame
        sns.kdeplot(data)
    else:
        for column in data.columns:
            sns.kdeplot(data, x = column)

    plt.xlim(right = 1)
    plt.ylim(top = 10)
    plt.title(f"Measurement distributions - {title}")
    plt.xlabel('Measurement value')
    plt.show()


def plot_distributions_KDE_combined(
        dfs: List[pd.DataFrame], title: str, metadata: dict
    ) -> None:
    """
    Same as plot_distributions_KDE, but first concatenates
    the dataframes in the list

    :param dfs: list of dataframes
    :param title: title of the plot
    :param metadata: metadata for the sensors
    """
    set_style()

    df = pd.concat(dfs)

    for column in df.columns:
        sns.kdeplot(df, x = column)

    plt.xlim(right = 1)
    plt.ylim(top = 10)
    plt.title(f"Measurement distributions - {title}")
    plt.xlabel('Measurement value')
    plt.show()
    

def plot_multiple_distributions(data: list, title: str, metadata: dict) -> None:
    """
    Plots the distribution of multiple sensors' measurements
    
    :param data: list of sensors' measurements
    :param title: title of the plot
    """
    set_style()
    
    # if isinstance(data1, pd.Series):    # distinguish between Series and DataFrame
    #     sns.kdeplot(data1, label = '1')
    #     sns.kdeplot(data2, label = '2')
    #     sns.kdeplot(data3, label = '3')
    # else:
        # for column in data1.columns:
        #     sns.kdeplot(data1, x = column, label = '1')
        # for column in data2.columns:
        #     sns.kdeplot(data2, x = column, label = '2')
        # for column in data3.columns:
        #     sns.kdeplot(data3, x = column, label = '3')
    for idx, df in enumerate(data):
        for column in df.columns:
            sns.kdeplot(df, x = column, label = idx + 1)

    plt.xlim(right = 1)
    plt.ylim(top = 10)
    plt.title(f"Measurement dist.s for {metadata['comp']} - {title}")
    plt.xlabel('Measurement value')
    plt.legend()
    plt.show()


def plot_tails(data: Union[pd.Series, pd.DataFrame], title: str) -> None:
    """
    Plots violin plot of a sensor's different component measurements
    
    Used source: https://seaborn.pydata.org/generated/seaborn.violinplot.html

    :param data: DataFrame or Series with component measurements
    :param title: plot title
    """
    set_style()

    if isinstance(data, pd.Series):     # interpret as Series
        df = pd.DataFrame(data).reset_index()
        df.columns = (['Component', 'DateTime', 'Value'])
        sns.violinplot(data = df, x = 'Component', y = 'Value', hue = 'Component', legend = False)
    else:
        df = data.reset_index()         # interpret as DataFrame
        df.columns = (['Component', 'DateTime', 'Value'])
        sns.violinplot(data = df, x = 'Component', y = 'Value', hue = 'Component', legend = False)

    plt.title(title)
    plt.ylim(top = 1.0)
    plt.xlabel('Component')
    plt.ylabel('Normalised value')
    plt.show()  