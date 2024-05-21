# Forecasting smog clouds with deep learning

This repository contains the scripts belonging to a paper titled "Forecasting Smog Clouds With Deep Learning: A Proof-Of-Concept", available at [INSERT].

For direct contact or questions:
v.w.oldenburg@student.rug.nl / vwoldenburg@gmail.com

## Description

The code available covers a pipeline from pollution and meteorological data (both made available by an initiative by the Dutch Government and the Royal Dutch Meteorological Institute (KNMI)) to using this data to forecast four constituents to smog over two locations in the Netherlands with a 1 hr lead time.

Main files here are:

- preprocess.py, which preprocesses the data; and
- model.py, which utilises the data and implements a hierarchical GRU model.

## Data

The source location was chosen to be in Utrecht, near the headquarters of the KNMI, and the target location is in Breukelen, both in the Netherlands. Data was sampled from years between 2017 and 2023 and obtained from:
- https://data.rivm.nl/data/ - for the pollution data; and
- https://dataplatform.knmi.nl/ - for the meteorological data.

The (raw) data is not uploaded to this repository, but can be added to the /data folders for the code to run.

## Dependencies:

The dependencies used in this project are listed in requirements.txt, though only "ordinary" libraries such as NumPy, pandas, and PyTorch are used.