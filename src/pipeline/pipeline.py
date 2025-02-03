# src/pipeline/pipeline.py

# This file contains one function that excecutes the entire pipeline.
# It is similar to the preprocess.ipynb notebook, but this one can be
# ran easily from the command line and main.py. It uses just functions
# from the pipeline modules, which are imported below.
# 
# For modifications, e.g. when changing the dimensions of time, space,
# and/or variables, using the notebook is probably easy for experimentation,
# and (copies of) this file would be good for "production" runs. So, to
# change the set-up, I'd recommend making a copy and going through the
# the code one-by-one, testing it, and then moving on to the next part

from pipeline import read_meteo_csv_from_data_raw
from pipeline import read_four_contaminants
from pipeline import get_metadata
from pipeline import tidy_raw_contaminant_data
from pipeline import tidy_raw_meteo_data
from pipeline import print_aggegrated_sensor_metrics
from pipeline import subset_sensors
from pipeline import perform_data_split
from pipeline import perform_data_split_without_train
from pipeline import print_split_ratios
from pipeline import calc_combined_min_max_params
from pipeline import normalise_linear
from pipeline import print_pollutant_extremes
from pipeline import export_minmax
from pipeline import plot_distributions_KDE
from pipeline import concat_frames_horizontally
from pipeline import delete_timezone_from_index
from pipeline import assert_equal_shape
from pipeline import assert_equal_index
from pipeline import assert_no_NaNs
from pipeline import assert_range


def execute_pipeline(
        contaminants: list = ['PM25', 'PM10', 'O3', 'NO2'],
        LOG: bool = True,
        SUBSET_MONTHS: bool = True,
        START_MON: str = '08',
        END_MON: str = '12',
        days_vali: int = 21,
        days_test: int = 21,
        days_vali_final_yrs: int = 63,
        days_test_final_yrs: int = 63
) -> None:
    """
    Executes the entire pipeline, from reading the raw data to
    normalising the data and exporting the minmax values. It does
    so in broadly the following steps:

    1. Read in the raw data
    2. Tidy the pollutant data
    3. Tidy the meteorological data
    4. Split the data
    5. Normalise the data
    6. Concatenate the data
    7. Export the data

    :param contaminants: list of contaminants to be read
    :param LOG: whether to print data transformation progress
    :param SUBSET_MONTHS: whether to subset the months (as opposed to
                          taking entire years of data). True gives
                          option for the following two booleans:
    :param START_MON: starting month for the subset(s)
    :param END_MON: ending month for the subset(s)
    :param days_vali: number of days in the validation set
    :param days_test: number of days in the test set
    :param days_vali_final_yrs: number of days in the validation set
                                for the final years
    :param days_test_final_yrs: number of days in the test set
                                for the final years
    :return: None
    """
    print('-----------------------------------')
    print('Executing the pipeline\n')

    # if PATH == None:
    #     raise ValueError('Please provide a path to the data...')

    # Sensor locations in the case of Utrecht area:
    DE_BILT = 'S260'       # starting (and only used) location for meteorological data
    TUINDORP = 'NL10636'   # starting location for contamination data
    BREUKELEN = 'NL10641'  # 'goal' location for contamination data

    # First step, load in the raw data
    df_PM25_2016_raw, df_PM10_2016_raw, df_O3_2016_raw, df_NO2_2016_raw = \
        read_four_contaminants(2016, contaminants)
    df_PM25_2017_raw, df_PM10_2017_raw, df_O3_2017_raw, df_NO2_2017_raw = \
        read_four_contaminants(2017, contaminants)
    df_PM25_2018_raw, df_PM10_2018_raw, df_O3_2018_raw, df_NO2_2018_raw = \
        read_four_contaminants(2018, contaminants)
    df_PM25_2019_raw, df_PM10_2019_raw, df_O3_2019_raw, df_NO2_2019_raw = \
        read_four_contaminants(2019, contaminants)
    df_PM25_2020_raw, df_PM10_2020_raw, df_O3_2020_raw, df_NO2_2020_raw = \
        read_four_contaminants(2020, contaminants)
    df_PM25_2021_raw, df_PM10_2021_raw, df_O3_2021_raw, df_NO2_2021_raw = \
        read_four_contaminants(2021, contaminants)
    df_PM25_2022_raw, df_PM10_2022_raw, df_O3_2022_raw, df_NO2_2022_raw = \
        read_four_contaminants(2022, contaminants)
    df_PM25_2023_raw, df_PM10_2023_raw, df_O3_2023_raw, df_NO2_2023_raw = \
        read_four_contaminants(2023, contaminants)


    df_meteo_2016_raw = read_meteo_csv_from_data_raw(2016)
    df_meteo_2017_raw = read_meteo_csv_from_data_raw(2017)
    df_meteo_2018_raw = read_meteo_csv_from_data_raw(2018)
    df_meteo_2019_raw = read_meteo_csv_from_data_raw(2019)
    df_meteo_2020_raw = read_meteo_csv_from_data_raw(2020)
    df_meteo_2021_raw = read_meteo_csv_from_data_raw(2021)
    df_meteo_2022_raw = read_meteo_csv_from_data_raw(2022)
    df_meteo_2023_raw = read_meteo_csv_from_data_raw(2023)


    if LOG:
        print('(1/8): Data read successfully')


    # First, tidy the contamination data

    PM25_2016_meta = get_metadata(df_PM25_2016_raw)
    PM10_2016_meta = get_metadata(df_PM10_2016_raw)
    O3_2016_meta   = get_metadata(df_O3_2016_raw)
    NO2_2016_meta  = get_metadata(df_NO2_2016_raw)
    PM25_2017_meta = get_metadata(df_PM25_2017_raw)
    PM10_2017_meta = get_metadata(df_PM10_2017_raw)
    O3_2017_meta   = get_metadata(df_O3_2017_raw)
    NO2_2017_meta  = get_metadata(df_NO2_2017_raw)
    PM25_2018_meta = get_metadata(df_PM25_2018_raw)
    PM10_2018_meta = get_metadata(df_PM10_2018_raw)
    O3_2018_meta   = get_metadata(df_O3_2018_raw)
    NO2_2018_meta  = get_metadata(df_NO2_2018_raw)
    PM25_2019_meta = get_metadata(df_PM25_2019_raw)
    PM10_2019_meta = get_metadata(df_PM10_2019_raw)
    O3_2019_meta   = get_metadata(df_O3_2019_raw)
    NO2_2019_meta  = get_metadata(df_NO2_2019_raw)
    PM25_2020_meta = get_metadata(df_PM25_2020_raw)
    PM10_2020_meta = get_metadata(df_PM10_2020_raw)
    O3_2020_meta   = get_metadata(df_O3_2020_raw)
    NO2_2020_meta  = get_metadata(df_NO2_2020_raw)
    PM25_2021_meta = get_metadata(df_PM25_2021_raw)
    PM10_2021_meta = get_metadata(df_PM10_2021_raw)
    O3_2021_meta   = get_metadata(df_O3_2021_raw)
    NO2_2021_meta  = get_metadata(df_NO2_2021_raw)
    PM25_2022_meta = get_metadata(df_PM25_2022_raw)
    PM10_2022_meta = get_metadata(df_PM10_2022_raw)
    O3_2022_meta   = get_metadata(df_O3_2022_raw)
    NO2_2022_meta  = get_metadata(df_NO2_2022_raw)
    PM25_2023_meta = get_metadata(df_PM25_2023_raw)
    PM10_2023_meta = get_metadata(df_PM10_2023_raw)
    O3_2023_meta   = get_metadata(df_O3_2023_raw)
    NO2_2023_meta  = get_metadata(df_NO2_2023_raw)

    df_PM25_2016_tidy = tidy_raw_contaminant_data(
        df_PM25_2016_raw, '2016', SUBSET_MONTHS, START_MON, END_MON)
    df_PM10_2016_tidy = tidy_raw_contaminant_data(
        df_PM10_2016_raw, '2016', SUBSET_MONTHS, START_MON, END_MON)
    df_O3_2016_tidy   = tidy_raw_contaminant_data(
        df_O3_2016_raw, '2016', SUBSET_MONTHS, START_MON, END_MON)
    df_NO2_2016_tidy  = tidy_raw_contaminant_data(
        df_NO2_2016_raw, '2016', SUBSET_MONTHS, START_MON, END_MON)
    df_PM25_2017_tidy = tidy_raw_contaminant_data(
        df_PM25_2017_raw, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_PM10_2017_tidy = tidy_raw_contaminant_data(
        df_PM10_2017_raw, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_O3_2017_tidy   = tidy_raw_contaminant_data(
        df_O3_2017_raw, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_NO2_2017_tidy  = tidy_raw_contaminant_data(
        df_NO2_2017_raw, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_PM25_2018_tidy = tidy_raw_contaminant_data(
        df_PM25_2018_raw, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_PM10_2018_tidy = tidy_raw_contaminant_data(
        df_PM10_2018_raw, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_O3_2018_tidy   = tidy_raw_contaminant_data(
        df_O3_2018_raw, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_NO2_2018_tidy  = tidy_raw_contaminant_data(
        df_NO2_2018_raw, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_PM25_2019_tidy = tidy_raw_contaminant_data(
        df_PM25_2019_raw, '2019', SUBSET_MONTHS, START_MON, END_MON)
    df_PM10_2019_tidy = tidy_raw_contaminant_data(
        df_PM10_2019_raw, '2019', SUBSET_MONTHS, START_MON, END_MON)
    df_O3_2019_tidy   = tidy_raw_contaminant_data(
        df_O3_2019_raw, '2019', SUBSET_MONTHS, START_MON, END_MON)
    df_NO2_2019_tidy  = tidy_raw_contaminant_data(
        df_NO2_2019_raw, '2019', SUBSET_MONTHS, START_MON, END_MON)
    df_PM25_2020_tidy = tidy_raw_contaminant_data(
        df_PM25_2020_raw, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_PM10_2020_tidy = tidy_raw_contaminant_data(
        df_PM10_2020_raw, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_O3_2020_tidy   = tidy_raw_contaminant_data(
        df_O3_2020_raw, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_NO2_2020_tidy  = tidy_raw_contaminant_data(
        df_NO2_2020_raw, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_PM25_2021_tidy = tidy_raw_contaminant_data(
        df_PM25_2021_raw, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_PM10_2021_tidy = tidy_raw_contaminant_data(
        df_PM10_2021_raw, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_O3_2021_tidy   = tidy_raw_contaminant_data(
        df_O3_2021_raw, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_NO2_2021_tidy  = tidy_raw_contaminant_data(
        df_NO2_2021_raw, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_PM25_2022_tidy = tidy_raw_contaminant_data(
        df_PM25_2022_raw, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_PM10_2022_tidy = tidy_raw_contaminant_data(
        df_PM10_2022_raw, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_O3_2022_tidy   = tidy_raw_contaminant_data(
        df_O3_2022_raw, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_NO2_2022_tidy  = tidy_raw_contaminant_data(
        df_NO2_2022_raw, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_PM25_2023_tidy = tidy_raw_contaminant_data(
        df_PM25_2023_raw, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_PM10_2023_tidy = tidy_raw_contaminant_data(
        df_PM10_2023_raw, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_O3_2023_tidy   = tidy_raw_contaminant_data(
        df_O3_2023_raw, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_NO2_2023_tidy  = tidy_raw_contaminant_data(
        df_NO2_2023_raw, '2023', SUBSET_MONTHS, START_MON, END_MON)


    # # print(df_PM25_2016_tidy.shape)
    # # print(df_PM10_2016_tidy.shape)
    # # print(df_O3_2016_tidy.shape)
    # # print(df_NO2_2016_tidy.shape)
    # print(df_PM25_2017_tidy.shape)
    # print(df_PM10_2017_tidy.shape)
    # print(df_O3_2017_tidy.shape)
    # print(df_NO2_2017_tidy.shape)
    # print(df_PM25_2018_tidy.shape)
    # print(df_PM10_2018_tidy.shape)
    # print(df_O3_2018_tidy.shape)
    # print(df_NO2_2018_tidy.shape)
    # # print(df_PM25_2019_tidy.shape)
    # # print(df_PM10_2019_tidy.shape)
    # # print(df_O3_2019_tidy.shape)
    # # print(df_NO2_2019_tidy.shape)
    # print(df_PM25_2020_tidy.shape)
    # print(df_PM10_2020_tidy.shape)
    # print(df_O3_2020_tidy.shape)
    # print(df_NO2_2020_tidy.shape)
    # print(df_PM25_2021_tidy.shape)
    # print(df_PM10_2021_tidy.shape)
    # print(df_O3_2021_tidy.shape)
    # print(df_NO2_2021_tidy.shape)
    # print(df_PM25_2022_tidy.shape)
    # print(df_PM10_2022_tidy.shape)
    # print(df_O3_2022_tidy.shape)
    # print(df_NO2_2022_tidy.shape)
    # print(df_PM25_2023_tidy.shape)
    # print(df_PM10_2023_tidy.shape)
    # print(df_O3_2023_tidy.shape)
    # print(df_NO2_2023_tidy.shape)


    if LOG:
        assert_equal_shape([
            # df_PM25_2016_tidy, df_PM10_2016_tidy, df_O3_2016_tidy, df_NO2_2016_tidy,
            df_PM25_2017_tidy, df_PM10_2017_tidy, df_O3_2017_tidy, df_NO2_2017_tidy,
            df_PM25_2018_tidy, df_PM10_2018_tidy, df_O3_2018_tidy, df_NO2_2018_tidy,
            # df_PM25_2019_tidy, df_PM10_2019_tidy, df_O3_2019_tidy, df_NO2_2019_tidy,
            df_PM25_2020_tidy, df_PM10_2020_tidy, df_O3_2020_tidy, df_NO2_2020_tidy,
            df_PM25_2021_tidy, df_PM10_2021_tidy, df_O3_2021_tidy, df_NO2_2021_tidy,
            df_PM25_2022_tidy, df_PM10_2022_tidy, df_O3_2022_tidy, df_NO2_2022_tidy,
            df_PM25_2023_tidy, df_PM10_2023_tidy, df_O3_2023_tidy, df_NO2_2023_tidy
            # Check for equal row length, not column length (there are a variable amount of
            # locations that measure each components, so column number is unequal)
        ], True, False, 'Tidying of pollutant data')
        print('(2/8): Pollutant data tidied successfully')


    # Second, tidy the meteorological data

    only_DeBilt = True                      # True: only De Bilt is used
    # # 2016
    # df_temp_2016_tidy = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'T', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_dewP_2016_tidy = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'TD', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_WD_2016_tidy   = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'DD', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_Wvh_2016_tidy  = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'FH', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_Wmax_2016_tidy = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'FX', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_preT_2016_tidy = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'DR', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_P_2016_tidy    = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'P', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_preS_2016_tidy = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'RH', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_SQ_2016_tidy   = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'SQ', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # df_Q_2016_tidy    = tidy_raw_meteo_data(
    #     df_meteo_2016_raw, 'Q', only_DeBilt, SUBSET_MONTHS, START_MON, END_MON)
    # 2017
    df_temp_2017_tidy = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'T', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_dewP_2017_tidy = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'TD', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_WD_2017_tidy   = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'DD', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_Wvh_2017_tidy  = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'FH', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_Wmax_2017_tidy = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'FX', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_preT_2017_tidy = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'DR', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_P_2017_tidy    = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'P', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_preS_2017_tidy = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'RH', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_SQ_2017_tidy   = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'SQ', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    df_Q_2017_tidy    = tidy_raw_meteo_data(
        df_meteo_2017_raw, 'Q', only_DeBilt, '2017', SUBSET_MONTHS, START_MON, END_MON)
    # 2018
    df_temp_2018_tidy = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'T', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_dewP_2018_tidy = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'TD', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_WD_2018_tidy   = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'DD', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_Wvh_2018_tidy  = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'FH', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_Wmax_2018_tidy = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'FX', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_preT_2018_tidy = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'DR', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_P_2018_tidy    = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'P', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_preS_2018_tidy = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'RH', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_SQ_2018_tidy   = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'SQ', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    df_Q_2018_tidy    = tidy_raw_meteo_data(
        df_meteo_2018_raw, 'Q', only_DeBilt, '2018', SUBSET_MONTHS, START_MON, END_MON)
    # # 2019
    # df_temp_2019_tidy = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'T', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_dewP_2019_tidy = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'TD', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_WD_2019_tidy   = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'DD', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_Wvh_2019_tidy  = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'FH', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_Wmax_2019_tidy = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'FX', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_preT_2019_tidy = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'DR', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_P_2019_tidy    = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'P', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_preS_2019_tidy = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'RH', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_SQ_2019_tidy   = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'SQ', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # df_Q_2019_tidy    = tidy_raw_meteo_data(
    #     df_meteo_2019_raw, 'Q', only_DeBilt, '2019', SUBSET_MONTHS, START_MON, END_MON)
    # 2020
    df_temp_2020_tidy = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'T', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_dewP_2020_tidy = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'TD', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_WD_2020_tidy   = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'DD', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_Wvh_2020_tidy  = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'FH', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_Wmax_2020_tidy = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'FX', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_preT_2020_tidy = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'DR', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_P_2020_tidy    = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'P', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_preS_2020_tidy = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'RH', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_SQ_2020_tidy   = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'SQ', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    df_Q_2020_tidy    = tidy_raw_meteo_data(
        df_meteo_2020_raw, 'Q', only_DeBilt, '2020', SUBSET_MONTHS, START_MON, END_MON)
    # 2021
    df_temp_2021_tidy = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'T', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_dewP_2021_tidy = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'TD', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_WD_2021_tidy   = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'DD', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_Wvh_2021_tidy  = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'FH', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_Wmax_2021_tidy = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'FX', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_preT_2021_tidy = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'DR', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_P_2021_tidy    = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'P', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_preS_2021_tidy = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'RH', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_SQ_2021_tidy   = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'SQ', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    df_Q_2021_tidy    = tidy_raw_meteo_data(
        df_meteo_2021_raw, 'Q', only_DeBilt, '2021', SUBSET_MONTHS, START_MON, END_MON)
    # 2022
    df_temp_2022_tidy = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'T', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_dewP_2022_tidy = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'TD', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_WD_2022_tidy   = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'DD', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_Wvh_2022_tidy  = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'FH', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_Wmax_2022_tidy = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'FX', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_preT_2022_tidy = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'DR', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_P_2022_tidy    = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'P', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_preS_2022_tidy = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'RH', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_SQ_2022_tidy   = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'SQ', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    df_Q_2022_tidy    = tidy_raw_meteo_data(
        df_meteo_2022_raw, 'Q', only_DeBilt, '2022', SUBSET_MONTHS, START_MON, END_MON)
    # 2023
    df_temp_2023_tidy = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'T', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_dewP_2023_tidy = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'TD', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_WD_2023_tidy   = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'DD', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_Wvh_2023_tidy  = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'FH', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_Wmax_2023_tidy = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'FX', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_preT_2023_tidy = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'DR', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_P_2023_tidy    = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'P', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_preS_2023_tidy = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'RH', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_SQ_2023_tidy   = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'SQ', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)
    df_Q_2023_tidy    = tidy_raw_meteo_data(
        df_meteo_2023_raw, 'Q', only_DeBilt, '2023', SUBSET_MONTHS, START_MON, END_MON)


    # print(df_temp_2017_tidy.shape)
    # print(df_dewP_2017_tidy.shape)
    # print(df_WD_2017_tidy.shape)
    # print(df_Wvh_2017_tidy.shape)
    # print(df_Wmax_2017_tidy.shape)
    # print(df_preT_2017_tidy.shape)
    # print(df_P_2017_tidy.shape)
    # print(df_preS_2017_tidy.shape)
    # print(df_SQ_2017_tidy.shape)
    # print(df_Q_2017_tidy.shape)

    # print(df_temp_2023_tidy.shape)
    # print(df_dewP_2023_tidy.shape)
    # print(df_WD_2023_tidy.shape)
    # print(df_Wvh_2023_tidy.shape)
    # print(df_Wmax_2023_tidy.shape)
    # print(df_preT_2023_tidy.shape)
    # print(df_P_2023_tidy.shape)
    # print(df_preS_2023_tidy.shape)
    # print(df_SQ_2023_tidy.shape)
    # print(df_Q_2023_tidy.shape)


    if LOG:
        assert_equal_shape([
            # df_temp_2016_tidy, df_dewP_2016_tidy, df_WD_2016_tidy, df_Wvh_2016_tidy, df_Wmax_2016_tidy,
            # df_preT_2016_tidy, df_P_2016_tidy, df_preS_2016_tidy, df_SQ_2016_tidy, df_Q_2016_tidy,
            df_temp_2017_tidy, df_dewP_2017_tidy, df_WD_2017_tidy, df_Wvh_2017_tidy, df_Wmax_2017_tidy,
            df_preT_2017_tidy, df_P_2017_tidy, df_preS_2017_tidy, df_SQ_2017_tidy, df_Q_2017_tidy,
            df_temp_2018_tidy, df_dewP_2018_tidy, df_WD_2018_tidy, df_Wvh_2018_tidy, df_Wmax_2018_tidy,
            df_preT_2018_tidy, df_P_2018_tidy, df_preS_2018_tidy, df_SQ_2018_tidy, df_Q_2018_tidy,
            # df_temp_2019_tidy, df_dewP_2019_tidy, df_WD_2019_tidy, df_Wvh_2019_tidy, df_Wmax_2019_tidy,
            # df_preT_2019_tidy, df_P_2019_tidy, df_preS_2019_tidy, df_SQ_2019_tidy, df_Q_2019_tidy,
            df_temp_2020_tidy, df_dewP_2020_tidy, df_WD_2020_tidy, df_Wvh_2020_tidy, df_Wmax_2020_tidy,
            df_preT_2020_tidy, df_P_2020_tidy, df_preS_2020_tidy, df_SQ_2020_tidy, df_Q_2020_tidy,
            df_temp_2021_tidy, df_dewP_2021_tidy, df_WD_2021_tidy, df_Wvh_2021_tidy, df_Wmax_2021_tidy,
            df_preT_2021_tidy, df_P_2021_tidy, df_preS_2021_tidy, df_SQ_2021_tidy, df_Q_2021_tidy,
            df_temp_2022_tidy, df_dewP_2022_tidy, df_WD_2022_tidy, df_Wvh_2022_tidy, df_Wmax_2022_tidy,
            df_preT_2022_tidy, df_P_2022_tidy, df_preS_2022_tidy, df_SQ_2022_tidy, df_Q_2022_tidy,
            df_temp_2023_tidy, df_dewP_2023_tidy, df_WD_2023_tidy, df_Wvh_2023_tidy, df_Wmax_2023_tidy,
            df_preT_2023_tidy, df_P_2023_tidy, df_preS_2023_tidy, df_SQ_2023_tidy, df_Q_2023_tidy
            # The meteorological tidying is done per component, so here we can check for
            # equal column length and equal row length (in contrast to the pollutant data)
        ], True, True, 'Tidying of meteorological data')
        # As can be seen in the raw data, the KNMI data should be complete and have no NaNs
        assert_no_NaNs([
            # df_temp_2016_tidy, df_dewP_2016_tidy, df_WD_2016_tidy, df_Wvh_2016_tidy, df_Wmax_2016_tidy,
            # df_preT_2016_tidy, df_P_2016_tidy, df_preS_2016_tidy, df_SQ_2016_tidy, df_Q_2016_tidy,
            df_temp_2017_tidy, df_dewP_2017_tidy, df_WD_2017_tidy, df_Wvh_2017_tidy, df_Wmax_2017_tidy,
            df_preT_2017_tidy, df_P_2017_tidy, df_preS_2017_tidy, df_SQ_2017_tidy, df_Q_2017_tidy,
            df_temp_2018_tidy, df_dewP_2018_tidy, df_WD_2018_tidy, df_Wvh_2018_tidy, df_Wmax_2018_tidy,
            df_preT_2018_tidy, df_P_2018_tidy, df_preS_2018_tidy, df_SQ_2018_tidy, df_Q_2018_tidy,
            # df_temp_2019_tidy, df_dewP_2019_tidy, df_WD_2019_tidy, df_Wvh_2019_tidy, df_Wmax_2019_tidy,
            # df_preT_2019_tidy, df_P_2019_tidy, df_preS_2019_tidy, df_SQ_2019_tidy, df_Q_2019_tidy,
            df_temp_2020_tidy, df_dewP_2020_tidy, df_WD_2020_tidy, df_Wvh_2020_tidy, df_Wmax_2020_tidy,
            df_preT_2020_tidy, df_P_2020_tidy, df_preS_2020_tidy, df_SQ_2020_tidy, df_Q_2020_tidy,
            df_temp_2021_tidy, df_dewP_2021_tidy, df_WD_2021_tidy, df_Wvh_2021_tidy, df_Wmax_2021_tidy,
            df_preT_2021_tidy, df_P_2021_tidy, df_preS_2021_tidy, df_SQ_2021_tidy, df_Q_2021_tidy,
            df_temp_2022_tidy, df_dewP_2022_tidy, df_WD_2022_tidy, df_Wvh_2022_tidy, df_Wmax_2022_tidy,
            df_preT_2022_tidy, df_P_2022_tidy, df_preS_2022_tidy, df_SQ_2022_tidy, df_Q_2022_tidy,
            df_temp_2023_tidy, df_dewP_2023_tidy, df_WD_2023_tidy, df_Wvh_2023_tidy, df_Wmax_2023_tidy,
            df_preT_2023_tidy, df_P_2023_tidy, df_preS_2023_tidy, df_SQ_2023_tidy, df_Q_2023_tidy
        ], 'Tydying of meteorological data')
        print('(3/8): Meteorological data tidied successfully')


    # print("Printing some basic statistics for the pollutants:")
    # print("(Sensor NL10636 is TUINDORP)\n")

    # print_aggegrated_sensor_metrics(
    #     [df_PM25_2017_tidy,
    #      df_PM25_2018_tidy,
    #      df_PM25_2020_tidy,
    #      df_PM25_2021_tidy,
    #      df_PM25_2022_tidy,
    #      df_PM25_2023_tidy], TUINDORP, PM25_2017_meta
    # )

    # print_aggegrated_sensor_metrics(
    #     [df_PM10_2017_tidy,
    #      df_PM10_2018_tidy,
    #      df_PM10_2020_tidy,
    #      df_PM10_2021_tidy,
    #      df_PM10_2022_tidy,
    #      df_PM10_2023_tidy], TUINDORP, PM10_2017_meta
    # )

    # print_aggegrated_sensor_metrics(
    #     [df_O3_2017_tidy,
    #      df_O3_2018_tidy,
    #      df_O3_2020_tidy,
    #      df_O3_2021_tidy,
    #      df_O3_2022_tidy,
    #      df_O3_2023_tidy], TUINDORP, O3_2017_meta
    # )

    # print_aggegrated_sensor_metrics(
    #     [df_NO2_2017_tidy,
    #      df_NO2_2018_tidy,
    #      df_NO2_2020_tidy,
    #      df_NO2_2021_tidy,
    #      df_NO2_2022_tidy,
    #      df_NO2_2023_tidy], TUINDORP, NO2_2017_meta
    # )


    del df_PM25_2016_raw, df_PM10_2016_raw, df_O3_2016_raw, df_NO2_2016_raw
    del df_PM25_2017_raw, df_PM10_2017_raw, df_O3_2017_raw, df_NO2_2017_raw
    del df_PM25_2018_raw, df_PM10_2018_raw, df_O3_2018_raw, df_NO2_2018_raw
    del df_PM25_2019_raw, df_PM10_2019_raw, df_O3_2019_raw, df_NO2_2019_raw
    del df_PM25_2020_raw, df_PM10_2020_raw, df_O3_2020_raw, df_NO2_2020_raw
    del df_PM25_2021_raw, df_PM10_2021_raw, df_O3_2021_raw, df_NO2_2021_raw
    del df_PM25_2022_raw, df_PM10_2022_raw, df_O3_2022_raw, df_NO2_2022_raw
    del df_PM25_2023_raw, df_PM10_2023_raw, df_O3_2023_raw, df_NO2_2023_raw
    del df_meteo_2016_raw
    del df_meteo_2017_raw
    del df_meteo_2018_raw
    del df_meteo_2019_raw
    del df_meteo_2020_raw
    del df_meteo_2021_raw
    del df_meteo_2022_raw
    del df_meteo_2023_raw


    # Here, we'll select the locations we want to use. The
    # I/O-task can be either 0-dimensional, or 1-dimensional.  

    # EDIT: The project is continued with a one-dimensional set-up,
    # but some code might still be accustomed to both possible set-ups.


    sensors_1D = [TUINDORP, BREUKELEN]

    df_PM25_2017_tidy_subset_1D = subset_sensors(df_PM25_2017_tidy, sensors_1D)
    df_PM10_2017_tidy_subset_1D = subset_sensors(df_PM10_2017_tidy, sensors_1D)
    df_O3_2017_tidy_subset_1D = subset_sensors(df_O3_2017_tidy, sensors_1D)
    df_NO2_2017_tidy_subset_1D = subset_sensors(df_NO2_2017_tidy, sensors_1D)
    df_PM25_2018_tidy_subset_1D = subset_sensors(df_PM25_2018_tidy, sensors_1D)
    df_PM10_2018_tidy_subset_1D = subset_sensors(df_PM10_2018_tidy, sensors_1D)
    df_O3_2018_tidy_subset_1D = subset_sensors(df_O3_2018_tidy, sensors_1D)
    df_NO2_2018_tidy_subset_1D = subset_sensors(df_NO2_2018_tidy, sensors_1D)
    df_PM25_2020_tidy_subset_1D = subset_sensors(df_PM25_2020_tidy, sensors_1D)
    df_PM10_2020_tidy_subset_1D = subset_sensors(df_PM10_2020_tidy, sensors_1D)
    df_O3_2020_tidy_subset_1D = subset_sensors(df_O3_2020_tidy, sensors_1D)
    df_NO2_2020_tidy_subset_1D = subset_sensors(df_NO2_2020_tidy, sensors_1D)
    df_PM25_2021_tidy_subset_1D = subset_sensors(df_PM25_2021_tidy, sensors_1D)
    df_PM10_2021_tidy_subset_1D = subset_sensors(df_PM10_2021_tidy, sensors_1D)
    df_O3_2021_tidy_subset_1D = subset_sensors(df_O3_2021_tidy, sensors_1D)
    df_NO2_2021_tidy_subset_1D = subset_sensors(df_NO2_2021_tidy, sensors_1D)
    df_PM25_2022_tidy_subset_1D = subset_sensors(df_PM25_2022_tidy, sensors_1D)
    df_PM10_2022_tidy_subset_1D = subset_sensors(df_PM10_2022_tidy, sensors_1D)
    df_O3_2022_tidy_subset_1D = subset_sensors(df_O3_2022_tidy, sensors_1D)
    df_NO2_2022_tidy_subset_1D = subset_sensors(df_NO2_2022_tidy, sensors_1D)
    df_PM25_2023_tidy_subset_1D = subset_sensors(df_PM25_2023_tidy, sensors_1D)
    df_PM10_2023_tidy_subset_1D = subset_sensors(df_PM10_2023_tidy, sensors_1D)
    df_O3_2023_tidy_subset_1D = subset_sensors(df_O3_2023_tidy, sensors_1D)
    df_NO2_2023_tidy_subset_1D = subset_sensors(df_NO2_2023_tidy, sensors_1D)

    del df_PM25_2017_tidy, df_PM10_2017_tidy, df_O3_2017_tidy, df_NO2_2017_tidy
    del df_PM25_2018_tidy, df_PM10_2018_tidy, df_O3_2018_tidy, df_NO2_2018_tidy
    del df_PM25_2020_tidy, df_PM10_2020_tidy, df_O3_2020_tidy, df_NO2_2020_tidy
    del df_PM25_2021_tidy, df_PM10_2021_tidy, df_O3_2021_tidy, df_NO2_2021_tidy
    del df_PM25_2022_tidy, df_PM10_2022_tidy, df_O3_2022_tidy, df_NO2_2022_tidy
    del df_PM25_2023_tidy, df_PM10_2023_tidy, df_O3_2023_tidy, df_NO2_2023_tidy


    # # print(df_NO2_2016_tidy_subset_1D.shape, df_O3_2016_tidy_subset_1D.shape,
    # #       df_PM25_2016_tidy_subset_1D.shape, df_PM10_2016_tidy_subset_1D.shape)
    # print(df_NO2_2017_tidy_subset_1D.shape, df_O3_2017_tidy_subset_1D.shape,
    #         df_PM25_2017_tidy_subset_1D.shape, df_PM10_2017_tidy_subset_1D.shape)
    # print(df_NO2_2018_tidy_subset_1D.shape, df_O3_2018_tidy_subset_1D.shape,
    #         df_PM25_2018_tidy_subset_1D.shape, df_PM10_2018_tidy_subset_1D.shape)
    # # print(df_NO2_2019_tidy_subset_1D.shape, df_O3_2019_tidy_subset_1D.shape,
    # #         df_PM25_2019_tidy_subset_1D.shape, df_PM10_2019_tidy_subset_1D.shape)
    # print(df_NO2_2020_tidy_subset_1D.shape, df_O3_2020_tidy_subset_1D.shape,
    #         df_PM25_2020_tidy_subset_1D.shape, df_PM10_2020_tidy_subset_1D.shape)
    # print(df_NO2_2021_tidy_subset_1D.shape, df_O3_2021_tidy_subset_1D.shape,
    #         df_PM25_2021_tidy_subset_1D.shape, df_PM10_2021_tidy_subset_1D.shape)
    # print(df_NO2_2022_tidy_subset_1D.shape, df_O3_2022_tidy_subset_1D.shape,
    #         df_PM25_2022_tidy_subset_1D.shape, df_PM10_2022_tidy_subset_1D.shape)

    # # # Add dummy column for missing cols
    # # df_O3_2016_tidy_subset_1D[TUINDORP] = np.nan
    # # df_O3_2019_tidy_subset_1D[TUINDORP] = np.nan


    if LOG:
        assert_equal_shape([
            # df_NO2_2016_tidy_subset_1D, df_O3_2016_tidy_subset_1D,
            # df_PM25_2016_tidy_subset_1D, df_PM10_2016_tidy_subset_1D,
            df_NO2_2017_tidy_subset_1D, df_O3_2017_tidy_subset_1D,
            df_PM25_2017_tidy_subset_1D, df_PM10_2017_tidy_subset_1D,
            df_NO2_2018_tidy_subset_1D, df_O3_2018_tidy_subset_1D,
            df_PM25_2018_tidy_subset_1D, df_PM10_2018_tidy_subset_1D,
            # df_NO2_2019_tidy_subset_1D, df_O3_2019_tidy_subset_1D,
            # df_PM25_2019_tidy_subset_1D, df_PM10_2019_tidy_subset_1D,
            df_NO2_2020_tidy_subset_1D, df_O3_2020_tidy_subset_1D,
            df_PM25_2020_tidy_subset_1D, df_PM10_2020_tidy_subset_1D,
            df_NO2_2021_tidy_subset_1D, df_O3_2021_tidy_subset_1D,
            df_PM25_2021_tidy_subset_1D, df_PM10_2021_tidy_subset_1D,
            df_NO2_2022_tidy_subset_1D, df_O3_2022_tidy_subset_1D,
            df_PM25_2022_tidy_subset_1D, df_PM10_2022_tidy_subset_1D
        # Check for both row and column length, as the data is now subsetted
        # for locations, and should have n_col of a year and n_row of x locations
        ], True, True, 'Location-wise subsetting of pollutant data')
        assert_no_NaNs([
            # df_NO2_2016_tidy_subset_1D, df_O3_2016_tidy_subset_1D,
            # df_PM25_2016_tidy_subset_1D, df_PM10_2016_tidy_subset_1D,
            df_NO2_2017_tidy_subset_1D, df_O3_2017_tidy_subset_1D,
            df_PM25_2017_tidy_subset_1D, df_PM10_2017_tidy_subset_1D,
            df_NO2_2018_tidy_subset_1D, df_O3_2018_tidy_subset_1D,
            df_PM25_2018_tidy_subset_1D, df_PM10_2018_tidy_subset_1D,
            # df_NO2_2019_tidy_subset_1D, df_O3_2019_tidy_subset_1D,
            # df_PM25_2019_tidy_subset_1D, df_PM10_2019_tidy_subset_1D,
            df_NO2_2020_tidy_subset_1D, df_O3_2020_tidy_subset_1D,
            df_PM25_2020_tidy_subset_1D, df_PM10_2020_tidy_subset_1D,
            df_NO2_2021_tidy_subset_1D, df_O3_2021_tidy_subset_1D,
            df_PM25_2021_tidy_subset_1D, df_PM10_2021_tidy_subset_1D,
            df_NO2_2022_tidy_subset_1D, df_O3_2022_tidy_subset_1D,
            df_PM25_2022_tidy_subset_1D, df_PM10_2022_tidy_subset_1D
        # By now, only locations with sufficient data should be left, and,
        # hence, no NaNs should be left in any of the pollutant dataframes
        ], 'Location-wise subsetting of pollutant data')
        print('(4/8): Location-wise subsetting of pollutant data successful')


    # Splitting the data into train, validation and test sets.
    # Each component is split separately. (All data remains
    # segregate for now for proper normalisation later.)

    df_PM25_2017_train_1D = df_PM25_2017_tidy_subset_1D.copy()
    df_PM10_2017_train_1D = df_PM10_2017_tidy_subset_1D.copy()
    df_NO2_2017_train_1D  = df_NO2_2017_tidy_subset_1D.copy()
    df_O3_2017_train_1D   = df_O3_2017_tidy_subset_1D.copy()
    df_temp_2017_train = df_temp_2017_tidy.copy()
    df_dewP_2017_train = df_dewP_2017_tidy.copy()
    df_WD_2017_train   = df_WD_2017_tidy.copy()
    df_Wvh_2017_train  = df_Wvh_2017_tidy.copy()
    df_P_2017_train    = df_P_2017_tidy.copy()
    df_SQ_2017_train   = df_SQ_2017_tidy.copy()

    df_PM25_2018_train_1D = df_PM25_2018_tidy_subset_1D.copy()
    df_PM10_2018_train_1D = df_PM10_2018_tidy_subset_1D.copy()
    df_NO2_2018_train_1D  = df_NO2_2018_tidy_subset_1D.copy()
    df_O3_2018_train_1D   = df_O3_2018_tidy_subset_1D.copy()
    df_temp_2018_train = df_temp_2018_tidy.copy()
    df_dewP_2018_train = df_dewP_2018_tidy.copy()
    df_WD_2018_train   = df_WD_2018_tidy.copy()
    df_Wvh_2018_train  = df_Wvh_2018_tidy.copy()
    df_P_2018_train    = df_P_2018_tidy.copy()
    df_SQ_2018_train   = df_SQ_2018_tidy.copy()

    df_PM25_2020_train_1D = df_PM25_2020_tidy_subset_1D.copy()
    df_PM10_2020_train_1D = df_PM10_2020_tidy_subset_1D.copy()
    df_NO2_2020_train_1D  = df_NO2_2020_tidy_subset_1D.copy()
    df_O3_2020_train_1D   = df_O3_2020_tidy_subset_1D.copy()
    df_temp_2020_train = df_temp_2020_tidy.copy()
    df_dewP_2020_train = df_dewP_2020_tidy.copy()
    df_WD_2020_train   = df_WD_2020_tidy.copy()
    df_Wvh_2020_train  = df_Wvh_2020_tidy.copy()
    df_P_2020_train    = df_P_2020_tidy.copy()
    df_SQ_2020_train   = df_SQ_2020_tidy.copy()

    df_PM25_2021_train_1D, df_PM25_2021_val_1D, df_PM25_2021_test_1D = \
        perform_data_split(df_PM25_2021_tidy_subset_1D, days_vali, days_test)
    df_PM10_2021_train_1D, df_PM10_2021_val_1D, df_PM10_2021_test_1D = \
        perform_data_split(df_PM10_2021_tidy_subset_1D, days_vali, days_test)
    df_NO2_2021_train_1D,  df_NO2_2021_val_1D,  df_NO2_2021_test_1D  = \
        perform_data_split(df_NO2_2021_tidy_subset_1D, days_vali, days_test)
    df_O3_2021_train_1D,   df_O3_2021_val_1D,   df_O3_2021_test_1D   = \
        perform_data_split(df_O3_2021_tidy_subset_1D, days_vali, days_test)
    df_temp_2021_train, df_temp_2021_val, df_temp_2021_test = \
        perform_data_split(df_temp_2021_tidy, days_vali, days_test)
    df_dewP_2021_train, df_dewP_2021_val, df_dewP_2021_test = \
        perform_data_split(df_dewP_2021_tidy, days_vali, days_test)
    df_WD_2021_train,   df_WD_2021_val,   df_WD_2021_test   = \
        perform_data_split(df_WD_2021_tidy, days_vali, days_test)
    df_Wvh_2021_train,  df_Wvh_2021_val,  df_Wvh_2021_test  = \
        perform_data_split(df_Wvh_2021_tidy, days_vali, days_test)
    df_P_2021_train,    df_P_2021_val,    df_P_2021_test    = \
        perform_data_split(df_P_2021_tidy, days_vali, days_test)
    df_SQ_2021_train,   df_SQ_2021_val,   df_SQ_2021_test   = \
        perform_data_split(df_SQ_2021_tidy, days_vali, days_test)

    df_PM25_2022_train_1D, df_PM25_2022_val_1D, df_PM25_2022_test_1D = \
        perform_data_split(df_PM25_2022_tidy_subset_1D, days_vali, days_test)
    df_PM10_2022_train_1D, df_PM10_2022_val_1D, df_PM10_2022_test_1D = \
        perform_data_split(df_PM10_2022_tidy_subset_1D, days_vali, days_test)
    df_NO2_2022_train_1D,  df_NO2_2022_val_1D,  df_NO2_2022_test_1D  = \
        perform_data_split(df_NO2_2022_tidy_subset_1D, days_vali, days_test)
    df_O3_2022_train_1D,   df_O3_2022_val_1D,   df_O3_2022_test_1D   = \
        perform_data_split(df_O3_2022_tidy_subset_1D, days_vali, days_test)
    df_temp_2022_train, df_temp_2022_val, df_temp_2022_test = \
        perform_data_split(df_temp_2022_tidy, days_vali, days_test)
    df_dewP_2022_train, df_dewP_2022_val, df_dewP_2022_test = \
        perform_data_split(df_dewP_2022_tidy, days_vali, days_test)
    df_WD_2022_train,   df_WD_2022_val,   df_WD_2022_test   = \
        perform_data_split(df_WD_2022_tidy, days_vali, days_test)
    df_Wvh_2022_train,  df_Wvh_2022_val,  df_Wvh_2022_test  = \
        perform_data_split(df_Wvh_2022_tidy, days_vali, days_test)
    df_P_2022_train,    df_P_2022_val,    df_P_2022_test    = \
        perform_data_split(df_P_2022_tidy, days_vali, days_test)
    df_SQ_2022_train,   df_SQ_2022_val,   df_SQ_2022_test   = \
        perform_data_split(df_SQ_2022_tidy, days_vali, days_test)

    df_PM25_2023_val_1D, df_PM25_2023_test_1D = \
        perform_data_split_without_train(
            df_PM25_2023_tidy_subset_1D, days_vali_final_yrs, days_test_final_yrs)
    df_PM10_2023_val_1D, df_PM10_2023_test_1D = \
        perform_data_split_without_train(
            df_PM10_2023_tidy_subset_1D, days_vali_final_yrs, days_test_final_yrs)
    df_NO2_2023_val_1D,  df_NO2_2023_test_1D  = \
        perform_data_split_without_train(
            df_NO2_2023_tidy_subset_1D, days_vali_final_yrs, days_test_final_yrs)
    df_O3_2023_val_1D,   df_O3_2023_test_1D   = \
        perform_data_split_without_train(
            df_O3_2023_tidy_subset_1D, days_vali_final_yrs, days_test_final_yrs)
    df_temp_2023_val,    df_temp_2023_test = \
        perform_data_split_without_train(
            df_temp_2023_tidy, days_vali_final_yrs, days_test_final_yrs)
    df_dewP_2023_val,    df_dewP_2023_test = \
        perform_data_split_without_train(
            df_dewP_2023_tidy, days_vali_final_yrs, days_test_final_yrs)
    df_WD_2023_val,      df_WD_2023_test   = \
        perform_data_split_without_train(
            df_WD_2023_tidy, days_vali_final_yrs, days_test_final_yrs)
    df_Wvh_2023_val,     df_Wvh_2023_test  = \
        perform_data_split_without_train(
            df_Wvh_2023_tidy, days_vali_final_yrs, days_test_final_yrs)
    df_P_2023_val,       df_P_2023_test    = \
        perform_data_split_without_train(
            df_P_2023_tidy, days_vali_final_yrs, days_test_final_yrs)
    df_SQ_2023_val,      df_SQ_2023_test   = \
        perform_data_split_without_train(
            df_SQ_2023_tidy, days_vali_final_yrs, days_test_final_yrs)


    if LOG:
        # First, check for equal shape of pollutant data of unsplitted years
        assert_equal_shape([
            df_PM25_2017_train_1D, df_PM10_2017_train_1D,
            df_NO2_2017_train_1D, df_O3_2017_train_1D,
            df_PM25_2018_train_1D, df_PM10_2018_train_1D,
            df_NO2_2018_train_1D, df_O3_2018_train_1D,
            df_PM25_2020_train_1D, df_PM10_2020_train_1D,
            df_NO2_2020_train_1D, df_O3_2020_train_1D
        ], True, True, 'Split of pollutant train set for 2017, 2018 and 2020')
        # Second, check for equal shape of meteorological data of unsplitted years
        assert_equal_shape([
            df_temp_2017_train, df_dewP_2017_train, df_WD_2017_train,
            df_Wvh_2017_train, df_P_2017_train, df_SQ_2017_train,
            df_temp_2018_train, df_dewP_2018_train, df_WD_2018_train,
            df_Wvh_2018_train, df_P_2018_train, df_SQ_2018_train,
            df_temp_2020_train, df_dewP_2020_train, df_WD_2020_train,
            df_Wvh_2020_train, df_P_2020_train, df_SQ_2020_train
        ], True, True, 'Split of meteorological train set for 2017, 2018 and 2020')
        # Third, check for equal row number of training set in 2021 and 2022
        assert_equal_shape([
            df_PM25_2021_train_1D, df_PM10_2021_train_1D,
            df_NO2_2021_train_1D, df_O3_2021_train_1D,
            df_temp_2021_train, df_dewP_2021_train, df_WD_2021_train,
            df_Wvh_2021_train, df_P_2021_train, df_SQ_2021_train,
            df_PM25_2022_train_1D, df_PM10_2022_train_1D,
            df_NO2_2022_train_1D, df_O3_2022_train_1D,
            df_temp_2022_train, df_dewP_2022_train, df_WD_2022_train,
            df_Wvh_2022_train, df_P_2022_train, df_SQ_2022_train
        # They should be of the same length, meaning they're split over the
        # same timeframe. Columns can vary, because meteorological data is
        # not used for the location where the predictions are made, i.e. Breukelen
        ], True, False, 'Split of training data for 2021 and 2022')
        # Fourth, check for equal row number of validation set in 2021 and 2022
        assert_equal_shape([
            df_PM25_2021_val_1D, df_PM10_2021_val_1D,
            df_NO2_2021_val_1D, df_O3_2021_val_1D,
            df_temp_2021_val, df_dewP_2021_val, df_WD_2021_val,
            df_Wvh_2021_val, df_P_2021_val, df_SQ_2021_val,
            df_PM25_2022_val_1D, df_PM10_2022_val_1D,
            df_NO2_2022_val_1D, df_O3_2022_val_1D,
            df_temp_2022_val, df_dewP_2022_val, df_WD_2022_val,
            df_Wvh_2022_val, df_P_2022_val, df_SQ_2022_val
        ], True, False, 'Split of validation data for 2021 and 2022')
        # Fifth, check for equal row number of test set in 2021 and 2022
        assert_equal_shape([
            df_PM25_2021_test_1D, df_PM10_2021_test_1D,
            df_NO2_2021_test_1D, df_O3_2021_test_1D,
            df_temp_2021_test, df_dewP_2021_test, df_WD_2021_test,
            df_Wvh_2021_test, df_P_2021_test, df_SQ_2021_test,
            df_PM25_2022_test_1D, df_PM10_2022_test_1D,
            df_NO2_2022_test_1D, df_O3_2022_test_1D,
            df_temp_2022_test, df_dewP_2022_test, df_WD_2022_test,
            df_Wvh_2022_test, df_P_2022_test, df_SQ_2022_test
        ], True, False, 'Split of test data for 2021 and 2022')
        # Sixth, check for equal row number of validation set in 2023
        assert_equal_shape([
            df_PM25_2023_val_1D, df_PM10_2023_val_1D,
            df_NO2_2023_val_1D, df_O3_2023_val_1D,
            df_temp_2023_val, df_dewP_2023_val, df_WD_2023_val,
            df_Wvh_2023_val, df_P_2023_val, df_SQ_2023_val
        ], True, False, 'Split of validation data for 2023')
        # Seventh, check for equal row number of test set in 2023
        assert_equal_shape([
            df_PM25_2023_test_1D, df_PM10_2023_test_1D,
            df_NO2_2023_test_1D, df_O3_2023_test_1D,
            df_temp_2023_test, df_dewP_2023_test, df_WD_2023_test,
            df_Wvh_2023_test, df_P_2023_test, df_SQ_2023_test
        ], True, False, 'Split of test data for 2023')
        print('(5/8): Train-validation-test split successful')


    print()
    print_split_ratios([df_PM25_2017_train_1D,
                        df_PM25_2018_train_1D,
                        df_PM25_2020_train_1D,
                        df_PM25_2021_train_1D,
                        df_PM25_2022_train_1D],
                        [df_PM25_2021_val_1D,
                        df_PM25_2022_val_1D,
                        df_PM25_2023_val_1D],
                        [df_PM25_2021_test_1D,
                        df_PM25_2022_test_1D,
                        df_PM25_2023_test_1D],
                        'the') # Could also print the pollutants here or any other string


    # Normalise each component separately, using the training data extremes


    PM25_min_train, PM25_max_train = calc_combined_min_max_params([
                                                                df_PM25_2017_train_1D,
                                                                df_PM25_2018_train_1D,
                                                                df_PM25_2020_train_1D,
                                                                df_PM25_2021_train_1D,
                                                                df_PM25_2022_train_1D,
                                                                ])
    PM10_min_train, PM10_max_train = calc_combined_min_max_params([
                                                                df_PM10_2017_train_1D,
                                                                df_PM10_2018_train_1D,
                                                                df_PM10_2020_train_1D,
                                                                df_PM10_2021_train_1D,
                                                                df_PM10_2022_train_1D,
                                                                ])
    O3_min_train,   O3_max_train   = calc_combined_min_max_params([
                                                                df_O3_2017_train_1D,
                                                                df_O3_2018_train_1D,
                                                                df_O3_2020_train_1D,
                                                                df_O3_2021_train_1D,
                                                                df_O3_2022_train_1D,
                                                                ])
    NO2_min_train,  NO2_max_train  = calc_combined_min_max_params([
                                                                df_NO2_2017_train_1D,
                                                                df_NO2_2018_train_1D,
                                                                df_NO2_2020_train_1D,
                                                                df_NO2_2021_train_1D,
                                                                df_NO2_2022_train_1D,
                                                                ])
    temp_min_train, temp_max_train = calc_combined_min_max_params([
                                                                df_temp_2017_train,
                                                                df_temp_2018_train,
                                                                df_temp_2020_train,
                                                                df_temp_2021_train,
                                                                df_temp_2022_train,
                                                                ])
    dewP_min_train, dewP_max_train = calc_combined_min_max_params([
                                                                df_dewP_2017_train,
                                                                df_dewP_2018_train,
                                                                df_dewP_2020_train,
                                                                df_dewP_2021_train,
                                                                df_dewP_2022_train,
                                                                ])
    WD_min_train,   WD_max_train   = calc_combined_min_max_params([
                                                                df_WD_2017_train,
                                                                df_WD_2018_train,
                                                                df_WD_2020_train,
                                                                df_WD_2021_train,
                                                                df_WD_2022_train,
                                                                ])
    Wvh_min_train,  Wvh_max_train  = calc_combined_min_max_params([
                                                                df_Wvh_2017_train,
                                                                df_Wvh_2018_train,
                                                                df_Wvh_2020_train,
                                                                df_Wvh_2021_train,
                                                                df_Wvh_2022_train,
                                                                ])
    P_min_train,    P_max_train    = calc_combined_min_max_params([
                                                                df_P_2017_train,
                                                                df_P_2018_train,
                                                                df_P_2020_train,
                                                                df_P_2021_train,
                                                                df_P_2022_train,
                                                                ])
    SQ_min_train,   SQ_max_train   = calc_combined_min_max_params([
                                                                df_SQ_2017_train,
                                                                df_SQ_2018_train,
                                                                df_SQ_2020_train,
                                                                df_SQ_2021_train,
                                                                df_SQ_2022_train,
                                                                ])

    print()
    df_minmax = print_pollutant_extremes(
        [NO2_min_train, NO2_max_train,
        O3_min_train, O3_max_train,
        PM10_min_train, PM10_max_train,
        PM25_min_train, PM25_max_train]
    )
    print()
    export_minmax(df_minmax, 'contaminant_minmax')


    df_NO2_2017_train_norm_1D = normalise_linear(df_NO2_2017_train_1D, NO2_min_train, NO2_max_train)
    df_NO2_2018_train_norm_1D = normalise_linear(df_NO2_2018_train_1D, NO2_min_train, NO2_max_train)
    df_NO2_2020_train_norm_1D = normalise_linear(df_NO2_2020_train_1D, NO2_min_train, NO2_max_train)
    df_NO2_2021_train_norm_1D = normalise_linear(df_NO2_2021_train_1D, NO2_min_train, NO2_max_train)
    df_NO2_2021_val_norm_1D = normalise_linear(df_NO2_2021_val_1D, NO2_min_train, NO2_max_train)
    df_NO2_2021_test_norm_1D = normalise_linear(df_NO2_2021_test_1D, NO2_min_train, NO2_max_train)
    df_NO2_2022_train_norm_1D = normalise_linear(df_NO2_2022_train_1D, NO2_min_train, NO2_max_train)
    df_NO2_val_2022_norm_1D = normalise_linear(df_NO2_2022_val_1D, NO2_min_train, NO2_max_train)
    df_NO2_test_2022_norm_1D = normalise_linear(df_NO2_2022_test_1D, NO2_min_train, NO2_max_train)
    df_NO2_val_2023_norm_1D = normalise_linear(df_NO2_2023_val_1D, NO2_min_train, NO2_max_train)
    df_NO2_test_2023_norm_1D = normalise_linear(df_NO2_2023_test_1D, NO2_min_train, NO2_max_train)

    df_O3_2017_train_norm_1D = normalise_linear(df_O3_2017_train_1D, O3_min_train, O3_max_train)
    df_O3_2018_train_norm_1D = normalise_linear(df_O3_2018_train_1D, O3_min_train, O3_max_train)
    df_O3_2020_train_norm_1D = normalise_linear(df_O3_2020_train_1D, O3_min_train, O3_max_train)
    df_O3_2021_train_norm_1D = normalise_linear(df_O3_2021_train_1D, O3_min_train, O3_max_train)
    df_O3_2021_val_norm_1D = normalise_linear(df_O3_2021_val_1D, O3_min_train, O3_max_train)
    df_O3_2021_test_norm_1D = normalise_linear(df_O3_2021_test_1D, O3_min_train, O3_max_train)
    df_O3_2022_train_norm_1D = normalise_linear(df_O3_2022_train_1D, O3_min_train, O3_max_train)
    df_O3_val_2022_norm_1D = normalise_linear(df_O3_2022_val_1D, O3_min_train, O3_max_train)
    df_O3_test_2022_norm_1D = normalise_linear(df_O3_2022_test_1D, O3_min_train, O3_max_train)
    df_O3_val_2023_norm_1D = normalise_linear(df_O3_2023_val_1D, O3_min_train, O3_max_train)
    df_O3_test_2023_norm_1D = normalise_linear(df_O3_2023_test_1D, O3_min_train, O3_max_train)

    df_PM10_2017_train_norm_1D = normalise_linear(df_PM10_2017_train_1D, PM10_min_train, PM10_max_train)
    df_PM10_2018_train_norm_1D = normalise_linear(df_PM10_2018_train_1D, PM10_min_train, PM10_max_train)
    df_PM10_2020_train_norm_1D = normalise_linear(df_PM10_2020_train_1D, PM10_min_train, PM10_max_train)
    df_PM10_2021_train_norm_1D = normalise_linear(df_PM10_2021_train_1D, PM10_min_train, PM10_max_train)
    df_PM10_2021_val_norm_1D = normalise_linear(df_PM10_2021_val_1D, PM10_min_train, PM10_max_train)
    df_PM10_2021_test_norm_1D = normalise_linear(df_PM10_2021_test_1D, PM10_min_train, PM10_max_train)
    df_PM10_2022_train_norm_1D = normalise_linear(df_PM10_2022_train_1D, PM10_min_train, PM10_max_train)
    df_PM10_val_2022_norm_1D = normalise_linear(df_PM10_2022_val_1D, PM10_min_train, PM10_max_train)
    df_PM10_test_2022_norm_1D = normalise_linear(df_PM10_2022_test_1D, PM10_min_train, PM10_max_train)
    df_PM10_val_2023_norm_1D = normalise_linear(df_PM10_2023_val_1D, PM10_min_train, PM10_max_train)
    df_PM10_test_2023_norm_1D = normalise_linear(df_PM10_2023_test_1D, PM10_min_train, PM10_max_train)

    df_PM25_2017_train_norm_1D = normalise_linear(df_PM25_2017_train_1D, PM25_min_train, PM25_max_train)
    df_PM25_2018_train_norm_1D = normalise_linear(df_PM25_2018_train_1D, PM25_min_train, PM25_max_train)
    df_PM25_2020_train_norm_1D = normalise_linear(df_PM25_2020_train_1D, PM25_min_train, PM25_max_train)
    df_PM25_2021_train_norm_1D = normalise_linear(df_PM25_2021_train_1D, PM25_min_train, PM25_max_train)
    df_PM25_2021_val_norm_1D = normalise_linear(df_PM25_2021_val_1D, PM25_min_train, PM25_max_train)
    df_PM25_2021_test_norm_1D = normalise_linear(df_PM25_2021_test_1D, PM25_min_train, PM25_max_train)
    df_PM25_2022_train_norm_1D = normalise_linear(df_PM25_2022_train_1D, PM25_min_train, PM25_max_train)
    df_PM25_val_2022_norm_1D = normalise_linear(df_PM25_2022_val_1D, PM25_min_train, PM25_max_train)
    df_PM25_test_2022_norm_1D = normalise_linear(df_PM25_2022_test_1D, PM25_min_train, PM25_max_train)
    df_PM25_val_2023_norm_1D = normalise_linear(df_PM25_2023_val_1D, PM25_min_train, PM25_max_train)
    df_PM25_test_2023_norm_1D = normalise_linear(df_PM25_2023_test_1D, PM25_min_train, PM25_max_train)

    df_temp_2017_train_norm = normalise_linear(df_temp_2017_train, temp_min_train, temp_max_train)
    df_temp_2018_train_norm = normalise_linear(df_temp_2018_train, temp_min_train, temp_max_train)
    df_temp_2020_train_norm = normalise_linear(df_temp_2020_train, temp_min_train, temp_max_train)
    df_temp_2021_train_norm = normalise_linear(df_temp_2021_train, temp_min_train, temp_max_train)
    df_temp_2021_val_norm = normalise_linear(df_temp_2021_val, temp_min_train, temp_max_train)
    df_temp_2021_test_norm = normalise_linear(df_temp_2021_test, temp_min_train, temp_max_train)
    df_temp_2022_train_norm = normalise_linear(df_temp_2022_train, temp_min_train, temp_max_train)
    df_temp_val_2022_norm = normalise_linear(df_temp_2022_val, temp_min_train, temp_max_train)
    df_temp_test_2022_norm = normalise_linear(df_temp_2022_test, temp_min_train, temp_max_train)
    df_temp_val_2023_norm = normalise_linear(df_temp_2023_val, temp_min_train, temp_max_train)
    df_temp_test_2023_norm = normalise_linear(df_temp_2023_test, temp_min_train, temp_max_train)

    df_dewP_2017_train_norm = normalise_linear(df_dewP_2017_train, dewP_min_train, dewP_max_train)
    df_dewP_2018_train_norm = normalise_linear(df_dewP_2018_train, dewP_min_train, dewP_max_train)
    df_dewP_2020_train_norm = normalise_linear(df_dewP_2020_train, dewP_min_train, dewP_max_train)
    df_dewP_2021_train_norm = normalise_linear(df_dewP_2021_train, dewP_min_train, dewP_max_train)
    df_dewP_2021_val_norm = normalise_linear(df_dewP_2021_val, dewP_min_train, dewP_max_train)
    df_dewP_2021_test_norm = normalise_linear(df_dewP_2021_test, dewP_min_train, dewP_max_train)
    df_dewP_2022_train_norm = normalise_linear(df_dewP_2022_train, dewP_min_train, dewP_max_train)
    df_dewP_val_2022_norm = normalise_linear(df_dewP_2022_val, dewP_min_train, dewP_max_train)
    df_dewP_test_2022_norm = normalise_linear(df_dewP_2022_test, dewP_min_train, dewP_max_train)
    df_dewP_val_2023_norm = normalise_linear(df_dewP_2023_val, dewP_min_train, dewP_max_train)
    df_dewP_test_2023_norm = normalise_linear(df_dewP_2023_test, dewP_min_train, dewP_max_train)

    df_WD_2017_train_norm = normalise_linear(df_WD_2017_train, WD_min_train, WD_max_train)
    df_WD_2018_train_norm = normalise_linear(df_WD_2018_train, WD_min_train, WD_max_train)
    df_WD_2020_train_norm = normalise_linear(df_WD_2020_train, WD_min_train, WD_max_train)
    df_WD_2021_train_norm = normalise_linear(df_WD_2021_train, WD_min_train, WD_max_train)
    df_WD_2021_val_norm = normalise_linear(df_WD_2021_val, WD_min_train, WD_max_train)
    df_WD_2021_test_norm = normalise_linear(df_WD_2021_test, WD_min_train, WD_max_train)
    df_WD_2022_train_norm = normalise_linear(df_WD_2022_train, WD_min_train, WD_max_train)
    df_WD_val_2022_norm = normalise_linear(df_WD_2022_val, WD_min_train, WD_max_train)
    df_WD_test_2022_norm = normalise_linear(df_WD_2022_test, WD_min_train, WD_max_train)
    df_WD_val_2023_norm = normalise_linear(df_WD_2023_val, WD_min_train, WD_max_train)
    df_WD_test_2023_norm = normalise_linear(df_WD_2023_test, WD_min_train, WD_max_train)

    df_Wvh_2017_train_norm = normalise_linear(df_Wvh_2017_train, Wvh_min_train, Wvh_max_train)
    df_Wvh_2018_train_norm = normalise_linear(df_Wvh_2018_train, Wvh_min_train, Wvh_max_train)
    df_Wvh_2020_train_norm = normalise_linear(df_Wvh_2020_train, Wvh_min_train, Wvh_max_train)
    df_Wvh_2021_train_norm = normalise_linear(df_Wvh_2021_train, Wvh_min_train, Wvh_max_train)
    df_Wvh_2021_val_norm = normalise_linear(df_Wvh_2021_val, Wvh_min_train, Wvh_max_train)
    df_Wvh_2021_test_norm = normalise_linear(df_Wvh_2021_test, Wvh_min_train, Wvh_max_train)
    df_Wvh_2022_train_norm = normalise_linear(df_Wvh_2022_train, Wvh_min_train, Wvh_max_train)
    df_Wvh_val_2022_norm = normalise_linear(df_Wvh_2022_val, Wvh_min_train, Wvh_max_train)
    df_Wvh_test_2022_norm = normalise_linear(df_Wvh_2022_test, Wvh_min_train, Wvh_max_train)
    df_Wvh_val_2023_norm = normalise_linear(df_Wvh_2023_val, Wvh_min_train, Wvh_max_train)
    df_Wvh_test_2023_norm = normalise_linear(df_Wvh_2023_test, Wvh_min_train, Wvh_max_train)

    df_P_2017_train_norm = normalise_linear(df_P_2017_train, P_min_train, P_max_train)
    df_P_2018_train_norm = normalise_linear(df_P_2018_train, P_min_train, P_max_train)
    df_P_2020_train_norm = normalise_linear(df_P_2020_train, P_min_train, P_max_train)
    df_P_2021_train_norm = normalise_linear(df_P_2021_train, P_min_train, P_max_train)
    df_P_2021_val_norm = normalise_linear(df_P_2021_val, P_min_train, P_max_train)
    df_P_2021_test_norm = normalise_linear(df_P_2021_test, P_min_train, P_max_train)
    df_P_2022_train_norm = normalise_linear(df_P_2022_train, P_min_train, P_max_train)
    df_P_val_2022_norm = normalise_linear(df_P_2022_val, P_min_train, P_max_train)
    df_P_test_2022_norm = normalise_linear(df_P_2022_test, P_min_train, P_max_train)
    df_P_val_2023_norm = normalise_linear(df_P_2023_val, P_min_train, P_max_train)
    df_P_test_2023_norm = normalise_linear(df_P_2023_test, P_min_train, P_max_train)

    df_SQ_2017_train_norm = normalise_linear(df_SQ_2017_train, SQ_min_train, SQ_max_train)
    df_SQ_2018_train_norm = normalise_linear(df_SQ_2018_train, SQ_min_train, SQ_max_train)
    df_SQ_2020_train_norm = normalise_linear(df_SQ_2020_train, SQ_min_train, SQ_max_train)
    df_SQ_2021_train_norm = normalise_linear(df_SQ_2021_train, SQ_min_train, SQ_max_train)
    df_SQ_2021_val_norm = normalise_linear(df_SQ_2021_val, SQ_min_train, SQ_max_train)
    df_SQ_2021_test_norm = normalise_linear(df_SQ_2021_test, SQ_min_train, SQ_max_train)
    df_SQ_2022_train_norm = normalise_linear(df_SQ_2022_train, SQ_min_train, SQ_max_train)
    df_SQ_val_2022_norm = normalise_linear(df_SQ_2022_val, SQ_min_train, SQ_max_train)
    df_SQ_test_2022_norm = normalise_linear(df_SQ_2022_test, SQ_min_train, SQ_max_train)
    df_SQ_val_2023_norm = normalise_linear(df_SQ_2023_val, SQ_min_train, SQ_max_train)
    df_SQ_test_2023_norm = normalise_linear(df_SQ_2023_test, SQ_min_train, SQ_max_train)


    if LOG:
        # Assert range only for training frames, validation and test
        # frames can, very theoretically, have unlimited values
        assert_range([
            df_NO2_2017_train_norm_1D, df_NO2_2018_train_norm_1D,
            df_NO2_2020_train_norm_1D, df_NO2_2021_train_norm_1D,
            df_NO2_2022_train_norm_1D
        ], 0, 1, 'Normalisation of NO2 data')
        assert_range([
            df_O3_2017_train_norm_1D, df_O3_2018_train_norm_1D,
            df_O3_2020_train_norm_1D, df_O3_2021_train_norm_1D,
            df_O3_2022_train_norm_1D
        ], 0, 1, 'Normalisation of O3 data')
        assert_range([
            df_PM10_2017_train_norm_1D, df_PM10_2018_train_norm_1D,
            df_PM10_2020_train_norm_1D, df_PM10_2021_train_norm_1D,
            df_PM10_2022_train_norm_1D
        ], 0, 1, 'Normalisation of PM10 data')
        assert_range([
            df_PM25_2017_train_norm_1D, df_PM25_2018_train_norm_1D,
            df_PM25_2020_train_norm_1D, df_PM25_2021_train_norm_1D,
            df_PM25_2022_train_norm_1D
        ], 0, 1, 'Normalisation of PM25 data')
        assert_range([
            df_temp_2017_train_norm, df_temp_2018_train_norm,
            df_temp_2020_train_norm, df_temp_2021_train_norm,
            df_temp_2022_train_norm
        ], 0, 1, 'Normalisation of temperature data')
        assert_range([
            df_dewP_2017_train_norm, df_dewP_2018_train_norm,
            df_dewP_2020_train_norm, df_dewP_2021_train_norm,
            df_dewP_2022_train_norm
        ], 0, 1, 'Normalisation of dew point data')
        assert_range([
            df_WD_2017_train_norm, df_WD_2018_train_norm,
            df_WD_2020_train_norm, df_WD_2021_train_norm,
            df_WD_2022_train_norm
        ], 0, 1, 'Normalisation of wind direction data')
        assert_range([
            df_Wvh_2017_train_norm, df_Wvh_2018_train_norm,
            df_Wvh_2020_train_norm, df_Wvh_2021_train_norm,
            df_Wvh_2022_train_norm
        ], 0, 1, 'Normalisation of wind velocity data')
        assert_range([
            df_P_2017_train_norm, df_P_2018_train_norm,
            df_P_2020_train_norm, df_P_2021_train_norm,
            df_P_2022_train_norm
        ], 0, 1, 'Normalisation of pressure data')
        assert_range([
            df_SQ_2017_train_norm, df_SQ_2018_train_norm,
            df_SQ_2020_train_norm, df_SQ_2021_train_norm,
            df_SQ_2022_train_norm
        ], 0, 1, 'Normalisation of solar radiation data')
        print('(6/8): Normalisation successful')


    # Now, create a big combined normalised dataframe for each year

    keys = ['PM25', 'PM10', 'O3', 'NO2',
            'temp', 'dewP', 'WD', 'Wvh', 'p', 'SQ']

    # Create input dataframes (u):
    # As we use the pollutant data twice, in Utrecht and Breukelen,
    # we add an index to sample only the Tuindorp (= Utrecht) data
    # for u, and later, we will add the Breukelen data for y
    frames_train_2017_1D_u = [df_PM25_2017_train_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_2017_train_norm_1D.loc[:, [TUINDORP]],
                                df_O3_2017_train_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_2017_train_norm_1D.loc[:, [TUINDORP]],
                                df_temp_2017_train_norm,
                                df_dewP_2017_train_norm,
                                df_WD_2017_train_norm,
                                df_Wvh_2017_train_norm,
                                df_P_2017_train_norm,
                                df_SQ_2017_train_norm]
    frames_train_2018_1D_u = [df_PM25_2018_train_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_2018_train_norm_1D.loc[:, [TUINDORP]],
                                df_O3_2018_train_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_2018_train_norm_1D.loc[:, [TUINDORP]],
                                df_temp_2018_train_norm,
                                df_dewP_2018_train_norm,
                                df_WD_2018_train_norm,
                                df_Wvh_2018_train_norm,
                                df_P_2018_train_norm,
                                df_SQ_2018_train_norm]
    frames_train_2020_1D_u = [df_PM25_2020_train_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_2020_train_norm_1D.loc[:, [TUINDORP]],
                                df_O3_2020_train_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_2020_train_norm_1D.loc[:, [TUINDORP]],
                                df_temp_2020_train_norm,
                                df_dewP_2020_train_norm,
                                df_WD_2020_train_norm,
                                df_Wvh_2020_train_norm,
                                df_P_2020_train_norm,
                                df_SQ_2020_train_norm]
    frames_train_2021_1D_u = [df_PM25_2021_train_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_2021_train_norm_1D.loc[:, [TUINDORP]],
                                df_O3_2021_train_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_2021_train_norm_1D.loc[:, [TUINDORP]],
                                df_temp_2021_train_norm,
                                df_dewP_2021_train_norm,
                                df_WD_2021_train_norm,
                                df_Wvh_2021_train_norm,
                                df_P_2021_train_norm,
                                df_SQ_2021_train_norm]
    frames_val_2021_1D_u = [df_PM25_2021_val_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_2021_val_norm_1D.loc[:, [TUINDORP]],
                                df_O3_2021_val_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_2021_val_norm_1D.loc[:, [TUINDORP]],
                                df_temp_2021_val_norm,
                                df_dewP_2021_val_norm,
                                df_WD_2021_val_norm,
                                df_Wvh_2021_val_norm,
                                df_P_2021_val_norm,
                                df_SQ_2021_val_norm]
    frames_test_2021_1D_u = [df_PM25_2021_test_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_2021_test_norm_1D.loc[:, [TUINDORP]],
                                df_O3_2021_test_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_2021_test_norm_1D.loc[:, [TUINDORP]],
                                df_temp_2021_test_norm,
                                df_dewP_2021_test_norm,
                                df_WD_2021_test_norm,
                                df_Wvh_2021_test_norm,
                                df_P_2021_test_norm,
                                df_SQ_2021_test_norm]
    frames_train_2022_1D_u = [df_PM25_2022_train_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_2022_train_norm_1D.loc[:, [TUINDORP]],
                                df_O3_2022_train_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_2022_train_norm_1D.loc[:, [TUINDORP]],
                                df_temp_2022_train_norm,
                                df_dewP_2022_train_norm,
                                df_WD_2022_train_norm,
                                df_Wvh_2022_train_norm,
                                df_P_2022_train_norm,
                                df_SQ_2022_train_norm]
    frames_val_2022_1D_u = [df_PM25_val_2022_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_val_2022_norm_1D.loc[:, [TUINDORP]],
                                df_O3_val_2022_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_val_2022_norm_1D.loc[:, [TUINDORP]],
                                df_temp_val_2022_norm,
                                df_dewP_val_2022_norm,
                                df_WD_val_2022_norm,
                                df_Wvh_val_2022_norm,
                                df_P_val_2022_norm,
                                df_SQ_val_2022_norm]
    frames_val_2023_1D_u = [df_PM25_val_2023_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_val_2023_norm_1D.loc[:, [TUINDORP]],
                                df_O3_val_2023_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_val_2023_norm_1D.loc[:, [TUINDORP]],
                                df_temp_val_2023_norm,
                                df_dewP_val_2023_norm,
                                df_WD_val_2023_norm,
                                df_Wvh_val_2023_norm,
                                df_P_val_2023_norm,
                                df_SQ_val_2023_norm]
    frames_test_2022_1D_u = [df_PM25_test_2022_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_test_2022_norm_1D.loc[:, [TUINDORP]],
                                df_O3_test_2022_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_test_2022_norm_1D.loc[:, [TUINDORP]],
                                df_temp_test_2022_norm,
                                df_dewP_test_2022_norm,
                                df_WD_test_2022_norm,
                                df_Wvh_test_2022_norm,
                                df_P_test_2022_norm,
                                df_SQ_test_2022_norm]
    frames_test_2023_1D_u = [df_PM25_test_2023_norm_1D.loc[:, [TUINDORP]],
                                df_PM10_test_2023_norm_1D.loc[:, [TUINDORP]],
                                df_O3_test_2023_norm_1D.loc[:, [TUINDORP]],
                                df_NO2_test_2023_norm_1D.loc[:, [TUINDORP]],
                                df_temp_test_2023_norm,
                                df_dewP_test_2023_norm,
                                df_WD_test_2023_norm,
                                df_Wvh_test_2023_norm,
                                df_P_test_2023_norm,
                                df_SQ_test_2023_norm]


    # For y, we only use pollutant data from Breukelen
    frames_train_2017_1D_y = [df_PM25_2017_train_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_2017_train_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_2017_train_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_2017_train_norm_1D.loc[:, [BREUKELEN]]]
    frames_train_2018_1D_y = [df_PM25_2018_train_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_2018_train_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_2018_train_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_2018_train_norm_1D.loc[:, [BREUKELEN]]]
    frames_train_2020_1D_y = [df_PM25_2020_train_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_2020_train_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_2020_train_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_2020_train_norm_1D.loc[:, [BREUKELEN]]]
    frames_train_2021_1D_y = [df_PM25_2021_train_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_2021_train_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_2021_train_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_2021_train_norm_1D.loc[:, [BREUKELEN]]]
    frames_val_2021_1D_y = [df_PM25_2021_val_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_2021_val_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_2021_val_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_2021_val_norm_1D.loc[:, [BREUKELEN]]]
    frames_test_2021_1D_y = [df_PM25_2021_test_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_2021_test_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_2021_test_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_2021_test_norm_1D.loc[:, [BREUKELEN]]]
    frames_train_2022_1D_y = [df_PM25_2022_train_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_2022_train_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_2022_train_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_2022_train_norm_1D.loc[:, [BREUKELEN]]]
    frames_val_2022_1D_y = [df_PM25_val_2022_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_val_2022_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_val_2022_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_val_2022_norm_1D.loc[:, [BREUKELEN]]]
    frames_val_2023_1D_y = [df_PM25_val_2023_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_val_2023_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_val_2023_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_val_2023_norm_1D.loc[:, [BREUKELEN]]]
    frames_test_2022_1D_y = [df_PM25_test_2022_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_test_2022_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_test_2022_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_test_2022_norm_1D.loc[:, [BREUKELEN]]]
    frames_test_2023_1D_y = [df_PM25_test_2023_norm_1D.loc[:, [BREUKELEN]],
                                df_PM10_test_2023_norm_1D.loc[:, [BREUKELEN]],
                                df_O3_test_2023_norm_1D.loc[:, [BREUKELEN]],
                                df_NO2_test_2023_norm_1D.loc[:, [BREUKELEN]]]


    input_keys = ['PM25', 'PM10', 'O3', 'NO2',
                'temp', 'dewP', 'WD', 'Wvh', 'p', 'SQ']
    target_keys = ['PM25', 'PM10', 'O3', 'NO2']


    # Now, we concatenate the dataframes horizontally
    df_train_2017_horizontal_u = concat_frames_horizontally(frames_train_2017_1D_u, input_keys)
    df_train_2018_horizontal_u = concat_frames_horizontally(frames_train_2018_1D_u, input_keys)
    df_train_2020_horizontal_u = concat_frames_horizontally(frames_train_2020_1D_u, input_keys)
    df_train_2021_horizontal_u = concat_frames_horizontally(frames_train_2021_1D_u, input_keys)
    df_val_2021_horizontal_u = concat_frames_horizontally(frames_val_2021_1D_u, input_keys)
    df_test_2021_horizontal_u = concat_frames_horizontally(frames_test_2021_1D_u, input_keys)
    df_train_2022_horizontal_u = concat_frames_horizontally(frames_train_2022_1D_u, input_keys)
    df_val_2022_horizontal_u = concat_frames_horizontally(frames_val_2022_1D_u, input_keys)
    df_val_2023_horizontal_u = concat_frames_horizontally(frames_val_2023_1D_u, input_keys)
    df_test_2022_horizontal_u = concat_frames_horizontally(frames_test_2022_1D_u, input_keys)
    df_test_2023_horizontal_u = concat_frames_horizontally(frames_test_2023_1D_u, input_keys)

    df_train_2017_horizontal_y = concat_frames_horizontally(frames_train_2017_1D_y, target_keys)
    df_train_2018_horizontal_y = concat_frames_horizontally(frames_train_2018_1D_y, target_keys)
    df_train_2020_horizontal_y = concat_frames_horizontally(frames_train_2020_1D_y, target_keys)
    df_train_2021_horizontal_y = concat_frames_horizontally(frames_train_2021_1D_y, target_keys)
    df_val_2021_horizontal_y = concat_frames_horizontally(frames_val_2021_1D_y, target_keys)
    df_test_2021_horizontal_y = concat_frames_horizontally(frames_test_2021_1D_y, target_keys)
    df_train_2022_horizontal_y = concat_frames_horizontally(frames_train_2022_1D_y, target_keys)
    df_val_2022_horizontal_y = concat_frames_horizontally(frames_val_2022_1D_y, target_keys)
    df_val_2023_horizontal_y = concat_frames_horizontally(frames_val_2023_1D_y, target_keys)
    df_test_2022_horizontal_y = concat_frames_horizontally(frames_test_2022_1D_y, target_keys)
    df_test_2023_horizontal_y = concat_frames_horizontally(frames_test_2023_1D_y, target_keys)

 
    # At last, a final check before exporting

    if LOG:
        # First, check if u-dataframes of unsplitted years have same shape
        assert_equal_shape([
            df_train_2017_horizontal_u, df_train_2018_horizontal_u,
            df_train_2020_horizontal_u,
        ], True, True, 'Shape of u-dataframes of 2017, 2018 and 2020')
        # Second, check if y-dataframes of unsplitted years have same shape
        assert_equal_shape([
            df_train_2017_horizontal_y, df_train_2018_horizontal_y,
            df_train_2020_horizontal_y,
        ], True, True, 'Shape of y-dataframes of 2017, 2018 and 2020')
        # Third, check if validation/test u-dataframes of splitted years
        # have the same shape
        assert_equal_shape([
            df_val_2021_horizontal_u, df_test_2021_horizontal_u,
            df_val_2022_horizontal_u, df_test_2022_horizontal_u,
        ], True, True, 'Shape of u-dataframes of 2021 and 2022')
        # Fourth, check if validation/test y-dataframes of splitted years
        # have the same shape
        assert_equal_shape([
            df_val_2021_horizontal_y, df_test_2021_horizontal_y,
            df_val_2022_horizontal_y, df_test_2022_horizontal_y,
        ], True, True, 'Shape of y-dataframes of 2021 and 2022')
        # Fifth, check if 2023 dataframes have the same shape
        assert_equal_shape([
            df_val_2023_horizontal_u, df_test_2023_horizontal_u,
            df_val_2023_horizontal_y, df_test_2023_horizontal_y,
        ], True, False, 'Shape of 2023 dataframes')
        
        print('(7/8): All data concatenations successful')


    # Save the dataframes to data_combined/ folder. The windowing will be performed
    # by a PyTorch Dataset class in the model scripts.

    df_train_2017_horizontal_u.to_csv("../data/data_combined/train_2017_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_train_2018_horizontal_u.to_csv("../data/data_combined/train_2018_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_train_2020_horizontal_u.to_csv("../data/data_combined/train_2020_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_train_2021_horizontal_u.to_csv("../data/data_combined/train_2021_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_val_2021_horizontal_u.to_csv("../data/data_combined/val_2021_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_test_2021_horizontal_u.to_csv("../data/data_combined/test_2021_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_train_2022_horizontal_u.to_csv("../data/data_combined/train_2022_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_val_2022_horizontal_u.to_csv("../data/data_combined/val_2022_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_val_2023_horizontal_u.to_csv("../data/data_combined/val_2023_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_test_2022_horizontal_u.to_csv("../data/data_combined/test_2022_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_test_2023_horizontal_u.to_csv("../data/data_combined/test_2023_combined_u.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')

    df_train_2017_horizontal_y.to_csv("../data/data_combined/train_2017_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_train_2018_horizontal_y.to_csv("../data/data_combined/train_2018_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_train_2020_horizontal_y.to_csv("../data/data_combined/train_2020_combined_y.csv",
    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_train_2021_horizontal_y.to_csv("../data/data_combined/train_2021_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_val_2021_horizontal_y.to_csv("../data/data_combined/val_2021_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_test_2021_horizontal_y.to_csv("../data/data_combined/test_2021_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_train_2022_horizontal_y.to_csv("../data/data_combined/train_2022_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_val_2022_horizontal_y.to_csv("../data/data_combined/val_2022_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_val_2023_horizontal_y.to_csv("../data/data_combined/val_2023_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_test_2022_horizontal_y.to_csv("../data/data_combined/test_2022_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')
    df_test_2023_horizontal_y.to_csv("../data/data_combined/test_2023_combined_y.csv",
                                    index = True, sep = ';', decimal = '.', encoding = 'utf-8')


    if LOG:
        print('(8/8): Data exported successfully')
        print('\nData preparation finished')
        print('-----------------------------------')