# src/preprocess.py

# main() for the preprocessing pipeline

from pipeline import execute_pipeline as run


def main():
    run(
        ['PM25', 'PM10', 'O3', 'NO2'],  # pass on contaminants;
                                        # for more variables, see pipeline/pipeline.py
    )


if __name__ == "__main__":
    main()