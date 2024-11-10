# src/pipeline/tidy.py

# Functions in this file transform the raw data from the KNMI and RIVM
# tidied dataframes. On top, functions for cleaning the contaminant data
# are defined and to the bottom more of the functions for cleaning the
# meteorological data. Two functions coordinate all these smaller functions:
# - 'tidy_raw_contaminant_data' for the pollutant data; and
# - 'tidy_raw_meteo_data' for the meteorological data.

import numpy as np
import pandas as pd


def col_contains_NaN(df: pd.DataFrame, col: str) -> bool:
    """
    Checks if column contains NaNs, returns True if so, False if not
    
    :param df: DataFrame
    :param col: column to check for NaNs
    :return: bool
    """
    return df[col].isna().any()


def get_component(df: pd.DataFrame) -> str:
    """
    Returns the component name of the contaminant by indexing the metadata

    :param df: DataFrame
    :return: str
    """
    return f"{df['Component'].iloc[0]}"


def get_unit(df: pd.DataFrame) -> str:
    """
    Returns the unit of the contaminant by indexing the metadata

    :param df: DataFrame
    :return: str
    """
    return f"{df['Eenheid'].iloc[0]}"


def get_metadata(df: pd.DataFrame) -> dict:
    """
    Returns dictionary with component and unit of contaminant, which
    can be used as attributary to the DataFrame containing the actual
    timeseries data, for e.g. plotting with the correct unit etc.
    
    :param df: DataFrame
    :return: dict
    """
    metadata = {'comp' : get_component(df),
                'unit' : get_unit(df)}
    return metadata


def remove_unuseful_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Removes columns from a DataFrame. As the name suggestsm, they
    are unuseful. This functions functionality is, however, not
    constrained to merely removing unuseful columns

    :param df: DataFrame
    :param cols: list of columns to remove
    :return: DataFrame without the specified columns
    """
    return df.drop(cols, axis = 1)


def change_contaminant_date_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes the date format to yyyy-mm-dd hh:mm, while doing some
    type checks. The column entries are converted to datetime objects
    and the column name is changed to 'DateTime' (and will later be used
    as the timeseries index).
    
    :param df: DataFrame
    :return: DataFrame with datetime column
    """
    try:
        df['Begindatumtijd'] = pd.to_datetime(df['Begindatumtijd'],
                                              format = '%Y%m%d %H:%M')
    except ValueError:
        df['Begindatumtijd'] = pd.to_datetime(df['Begindatumtijd'],
                                              format = 'ISO8601')
    df.rename(columns = {'Begindatumtijd' : 'DateTime'},
              inplace = True)
    return df


def strip_dot1_of_col_names(col_names: list) -> list: 
    """
    In the process of data acquisition at the KNMI and RIVM, some sensors
    experienced errors, and were restarted after a while (or repared, whatever).
    As a result, some sensors have two columns in the raw data, which are, during
    extraction, automatically named with a '.1' suffix. This function removes
    the '.1' suffix from the column names. (Later, the columns are merged.)

    :param col_names: list of column names
    :return: list of column names without the '.1' suffix
    """
    return [name.removesuffix('.1') for name in np.asarray(col_names)]


def resolve_split_columns(df: pd.DataFrame) -> pd.DataFrame:          
    """
    Groups cols which were split over two columns in the raw data.
    (See strip_dot1_of_col_names() for more info on the cause of this.)
    The operation is nontrivial and requires a few steps:
    - transpose the df
    - group by duplicate column names
    - sum the cols
    - transpose again

    :param df: DataFrame
    :return: DataFrame with split columns resolved
    """
    df.columns = strip_dot1_of_col_names(df.columns)

    # tranpose; group by duplicate column names; no sorting; sum the cols;
    # minimum count is 1 to get NaN when column is empty; transpose again
    return df.transpose().groupby(by = df.columns, sort = False).sum(min_count = 1).transpose()


def fill_NaNs_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Just one of the many ways to fill in NaNs: fills in NaNs by
    copying last value - forward fill. Later comment: because of
    its simplicity, however, this function is not actually used.

    :param df: DataFrame
    :return: DataFrame with NaNs filled in
    """
    return df.ffill(axis = 1)


def fill_NaNs_linear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills in NaNs by linear interpolation, with a maximum of a week (168 hours).
    When more data is missing, it's probably better to not use the data to avoid
    introducing too much bias. Hence, the limit of 168 hours.

    A possible TODO for later is to implement a more sophisticated method, e.g.:
    - polynomial fitting;
    - spline fitting;
    - kriging (through time-space kriging or treating time as a spatial dimension);
    but most promising (and still easy) is probably:
    - sinusoidal fitting, which can help capture periodicity caused by e.g. daily
    meteorological patterns (which influence the contaminant levels), daily commute,
    industrial activity etc., or weekly patterns (e.g. less traffic on weekends), when
    using multiple frequencies.
    
    :param df: DataFrame
    :return: DataFrame with NaNs filled in
    """
    return df.interpolate(method = 'linear', limit = 24 * 7)


def subset_month_range(
        df: pd.DataFrame, start_mon: str, end_mon: str, year: str
    ) -> pd.DataFrame:
    """
    Subsets a specified month range from the DataFrame

    :param df: DataFrame
    :param start_mon: starting month
    :param end_mon: ending month
    :param year: year
    :return: DataFrame with subsetted month range
    """
    try:
        start_date = f'{int(year)}-{int(start_mon):02d}'
        end_date = f'{int(year)}-{int(end_mon):02d}'
        return df[start_date : end_date]
    except ValueError as exc:
        print(f"Error: {exc}")
        print(f"Invalid date range: {start_date} to {end_date}")
        return df


def delete_feb_29th(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deletes the 29th of February from a df for YOY consistency

    :param df: DataFrame
    :return: DataFrame without the 29th of February
    """
    return df[~((df.index.month == 2) & (df.index.day == 29))]

def delete_firework_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deletes the 31st of December and 1st of January from the data,
    because the structurally have abnormally high contaminant levels.
    For a plotted example, see Appendix B.4 of the thesis.

    :param df: DataFrame
    :return: DataFrame without the 31st of December and 1st of January
    """
    return df[~(((df.index.month == 12) & (df.index.day == 31)) | \
                 ((df.index.month == 1) & (df.index.day == 1)))]


def delete_empty_columns(df: pd.DataFrame, NaN_max_percentage: int = 0.25) -> pd.DataFrame:
    """
    Drops cols with more than 25% NaNs. This can be varied
    by changing the threshold parameter

    :param df: DataFrame
    :param delete_threshold: NaN maximum for dropping columns
    :return: DataFrame without columns with more than 25% NaNs
    """          
    threshold = df.shape[0] * (1 - NaN_max_percentage)
    return df.dropna(thresh = threshold, axis = 1)


def make_index_timezone_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes the index timezone naive, because the timezone is not
    relevant for the data, and it's easier to work with the data
    when the timezone is removed. To be more specific: pollutant
    data from 2023 onward seems to be timezone aware, the rest not

    :param df: DataFrame
    :return: DataFrame with timezone naive index
    """
    return df.tz_localize(None)


def tidy_raw_contaminant_data(
        df: pd.DataFrame, year: str, subset_months: bool = True, 
        start_mon: str = '01', end_mon: str = '12', fill_NaNs: bool = True
    ) -> pd.DataFrame:
    """
    Tidies raw contaminant data by various preprocessing steps and 
    previously defined helper functions:
    - remove leading and trailing ws in col names;
    - remove unuseful cols;
    - change date format to yr-mm-dd hr-mn;
    - set the index to the dates ('datetime');
    - resolve split columns;
    - fill in NaNs using linear interpolation;
    - subset months;
    - delete outliers (29th of February, 31st of December and 1st of January);
    - drop columns which remained too empty after interpolating (interpolation
    had a maximum of 168 to prevent too much bias slipping in);
    and returns the cleaned DataFrame.

    :param df: DataFrame to clean
    :param year: year of the data
    :param subset_months: bool, whether to subset the months
    :param start_mon: starting month for the subset
    :param end_mon: ending month for the subset
    :param fill_NaNs: bool, whether to fill in NaNs
    :return: tidied DataFrame
    """
    df.columns = df.columns.str.strip() # remove leading and trailing ws in col names
    df = remove_unuseful_cols(df, ['Component', 'Bep.periode', 'Eenheid', 'Einddatumtijd'])
                                        # change format to yr-mm-dd hr-mn
    df = change_contaminant_date_format(df)         
                                        # set the index to the dates ('datetime')
    df = df.set_index('DateTime', drop = True)
    df = make_index_timezone_naive(df)  # make the index timezone naive
    df = resolve_split_columns(df)      # concat sensor data split over two columns

    if fill_NaNs:
        df = fill_NaNs_linear(df)       # fill in NaNs using linear interpolation
                                        
    if subset_months:
        df = subset_month_range(df, start_mon, end_mon, year)
    df = delete_feb_29th(df)            # delete the 29th of February
    df = delete_firework_days(df)       # delete the 31st of December and 1st of January
    df = delete_empty_columns(df)       # drop columns which remained too empty after interpolating

    # BEWARE: not all data is necessarily filled in by now, there could still be NaNs.
    # Later, when the data is also subsetted by sensors, only sensors with enough data
    # are chosen, "solving" this problem. Allowing some NaNs here helps a bit with looking
    # for adequate data, and the interpolation and filtering here is to just set a baseline
    # of eligibility for the data to be used in the model, as made as of now.

    return df


def change_meteo_date_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes the date format to yyyy-mm-dd hh:mm, specifically
    for the meteo data format
    
    :param df: the DataFrame to change the date format of
    :return: the DataFrame with the changed date format
    """
    df['DateTime'] = pd.to_datetime(df['YYYYMMDD'].astype(str) + ' ' + df['HH'].astype(str), 
                                    format = '%Y%m%d %H')
    df = remove_unuseful_cols(df, ['YYYYMMDD', 'HH'])
    return df


def replace_WD_990_with_NaN(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Replaces all occurrences of 990 in the WD column with 0:
    wind direction is represented in degrees, and 990 is used
    when the wind is calm or could not measured. As this would
    be a large outlier, it is replaced with just a 0
    
    :param df: the DataFrame to replace the values in
    :param col: the column to replace the values in
    :return: the DataFrame with the replaced values
    """
    df[col] = df[col].replace(990, np.nan)
    return df


def tidy_raw_meteo_data(
        df: pd.DataFrame, col: str, only_260: bool, year: str,
        subset_months, start_mon, end_mon, fill_NaNs = True):
    """
    Tidies the raw meteo data by various preprocessing steps:
    - removing leading and trailing whitespace in column names;
    - renaming the station column to 'STN';
    - removing unuseful columns (i.e. unused meteorological parameters);
    - subtracting 1 from the hour column to get a 0-23 hour range;
    - changing the date format to yyyy-mm-dd hh:mm;
    - setting the DateTime column as the index;
    - replacing 990 in the WD column with NaN;
    - continuing with only the 260 station (the one used in the thesis):
    - filling NaN values with linear interpolation;
    - subsetting the data to the specified months;
    - renaming the column to the station name;
    - deleting February 29th and firework days;
    - returning the 260 station data for the specified year and air component.

    Here, station 260 is the De Bilt station used in the thesis. For the code
    to potentially work with other stations, the station number should be changed
    together with some of the rest of the function calls to properly work.

    :param df: the DataFrame to tidy
    :param col: the column to keep in the DataFrame
    :param only_260: whether to return only the 260 station data
    :param year: the year of the data
    :param subset_months: whether to subset the months
    :param start_mon: the starting month for the subset
    :param end_mon: the ending month for the subset
    :param fill_NaNs: whether to fill NaN values with linear interpolation
    :return: the 260 station DataFrame
    """
    df.columns = df.columns.str.strip() # remove leading and trailing ws in col names
    if '# STN' in df.columns:           # change col name of stations
        df = df.rename(columns = {'# STN' : 'STN'})
    df = remove_unuseful_cols(df, ['T10N', 'FF', 'VV', 'N', 'U',
                                   'WW', 'IX', 'M', 'R', 'O', 'S', 'Y'])
    df['HH'] = df['HH'].subtract(1)     # 1-24 to 0-23 hour range
    df = change_meteo_date_format(df)   # create DateTime column
    df = df.set_index('DateTime')       # set DateTime as index
    df = make_index_timezone_naive(df)  # make the index timezone naive

    df = df[[col, 'STN']].copy()        # keep only selected col and station name col
    if col == 'DD':                     # 990 (change in DD (or WD)) -> 0, for more even influence
        df = replace_WD_990_with_NaN(df, col)

                                        # continue with the 260 station:
    df_260 = remove_unuseful_cols(df[df['STN'] == 260], 'STN')

    if fill_NaNs:
        df_260 = fill_NaNs_linear(df_260).astype('float64')

    if subset_months:
        df_260 = subset_month_range(df_260, start_mon, end_mon, year)

    df_260 = df_260.rename(columns = {df.columns[0] : 'S260'})
    df_260 = delete_feb_29th(df_260)
    df_260 = delete_firework_days(df_260)

    if only_260:                        # return only the 260 station (only_260 var is unused)
        return df_260
    else:
        return None