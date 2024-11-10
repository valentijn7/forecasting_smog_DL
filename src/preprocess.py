# src/preprocess.py

# main() for the preprocessing pipeline

from pipeline import execute_pipeline as run


def main():
    run(
        'tinus',                        # pass on device name
                                        # pass on path (current working directory)
        r"c:\Users\vwold\Documents\Bachelor\ICML_paper\forecasting_smog_DL\forecasting_smog_DL\src",
        ['PM25', 'PM10', 'O3', 'NO2'],  # pass on contaminants
        LOGGER = True,                  # pass on LOGGER
                                        # For more variables, see pipeline/pipeline.py
    )


if __name__ == "__main__":
    main()