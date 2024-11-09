# src/pipeline/transform.py

# More general functions that subset, aggregate, manipulate,
# or transform the data and datasets in some way during the pipeline

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
    

