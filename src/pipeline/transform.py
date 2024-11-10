# src/pipeline/transform.py

# More general functions that subset, aggregate, manipulate,
# or transform the data and datasets in some way during the pipeline

from typing import List, Dict
import pandas as pd


def subset_sensors(df, sensors):
    """
    "Subsets sensor in the vicinity of Groningen, Friesland, and Drenthe"
    This is what it was in the beginning of the project, but now it just
    takes a column or columns from the dataframe and returns it/them
    """
    if isinstance(sensors, str):        # subset one sensor, so a str,
        return df.loc[:, sensors]       
    else:                               # else, subset multiple from a list
        return df.loc[:, df.columns.isin(sensors)]
    

def concat_frames_vertically(dfs: List[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
    """
    Concatenates a list of dataframes into one dataframe with a MultiIndex,
    where the first level is the key and the second level is the original index.
    The values get unionized over the columns, and the index is sorted by date.

    A used source: https://pandas.pydata.org/docs/user_guide/cookbook.html

    UPDATE: This function is not used anymore, the data is now concatenated
    differently, see concat_frames_horizontally(), and the data sampling is
    performed later, during modelling, partly for (memory) efficiency reasons
    
    :param dfs: List of dataframes to concatenate
    :param keys: List of keys to use as the first level of the MultiIndex
    :return: Concatenated dataframe with MultiIndex, sorted on time
    """
    # All columns get the same name, because pd.concat() uses column names
    # to match columns in this case, and we want to unionize the columns
    frames = [df.rename(columns = {df.columns[0] : 'Groningen'}) for df in dfs]
    
    return pd.concat(objs = frames, 
                     axis = 0,          # concat over row axis while unionizing column-axis
                     join = 'outer',    # create MultiIndex, rename MultiIndex, then sort on date
                     keys = keys).rename_axis(['Component', 'DateTime']).sort_index(level = 'DateTime')


def make_manual_dict_of_dfs(
        dfs: List[pd.DataFrame], components: List[str]
    ) -> Dict[str, pd.DataFrame]:
    """
    Creates a dictionary of dataframes with the components as keys.
    Beware: it assumes the same order as the components list!

    :param dfs: list of dataframes
    :param components: list of components
    :return: created dictionary of dataframes
    """
    return dict(zip(components, dfs))


def sort_dict_of_dfs(
        dfs_dict: Dict[str, pd.DataFrame], components_sorted: List[str]
        ) -> List[pd.DataFrame]:
    """
    Sorts (or rearranges) a dictionary of dataframes
    by the given components list

    :param dfs_dict: dictionary of dataframes
    :param components_sorted: list of components
    :return: sorted list of dataframes
    """
    return [dfs_dict[name] for name in components_sorted]


def print_dict_of_dfs(dfs: List[pd.DataFrame]) -> None:
    """
    Prints a dictionary of dataframes in a concise manner (pprint)
    
    :param dfs: dictionary of dataframes
    """
    for key, df in dfs.items():
        print(key)
        print(df.head(2))


def print_dfs_sorted(dfs: List[pd.DataFrame]) -> None:
    """
    Prints a list of dataframes in a concise manner

    :param dfs: list of dataframes
    """
    for df in dfs:
        print(df.head(2))


def concat_frames_horizontally(
        dfs: List[pd.DataFrame], components: List[str]
        ) -> pd.DataFrame:
    """
    An important function, so here's a detailed explanation:
    
    Concatenates a list of dataframes into one dataframe where:
    - the x-axis (axis = 1) is the DateTime index;
    - the y-axis (axis = 0) is the component axis.
    With this approach, the diagrams cannot be automatically sorted.
    Hence, this is done manually through the sorting, or rearragement,
    order of the components list. In thesis, this was:
    - components = 
        'PM25', 'PM10', 'O3', 'NO2', 'temp', 'dewP', 'WD', 'Wvh', 'p', 'SQ'
    which then gets sorted into:
    - components_sorted =
        'NO2', 'O3', 'PM10', 'PM25', 'SQ', 'WD', 'Wvh', 'dewP', 'p', 'temp'
    For the output keys, this is the same but excluding the meteo params.

    In summary, the following steps are taken:
    1. Create a dictionary of dataframes with the components as keys;
    2. Sort the components list alphabetically;
    3. Sort the dictionary by the components list;
    4. Concatenate the sorted dataframes;
    5. Drop the old column names (i.e. sensor names);
    6. Return the concatenated dataframe.

    :param dfs: list of dataframes
    :param components: list of components
    :return: concatenated dataframe
    """
    dfs_dict = make_manual_dict_of_dfs(dfs, components)
    components_sorted = sorted(components)
    dfs_sorted = sort_dict_of_dfs(dfs_dict, components_sorted)
    
    df = pd.concat(objs = dfs_sorted, 
                   axis = 1,            # concat over column axis    
                   keys = components_sorted).sort_index(level = 'DateTime')
    df.columns = df.columns.droplevel(1)
    return df


def delete_timezone_from_index(dfs: List[pd.DataFrame]) -> None:
    """ 
    The 2023 data seems to have a different datetime type,
    one including timezone information. This function removes
    the timezone information from the index for uniformity
    and to avoid errors later on. If we were to add 2024 data
    later on, we should add a check for the timezone information
    again with, for example:

    for df in frames_val_2023_1D_u:
        print(df.index.tz)

    :param dfs: list of dataframes
    """
    for df in dfs:
        df.index = df.index.tz_localize(None)