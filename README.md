# UofG_ml_BTC_returns

University project about predicting Bitcoin (BTC) returns using ensemble machine learning models.

## Dataset

The project uses two datasets:

1. [BTC:USDT_price_1dfreq.csv](BTC:USDT_price_1dfreq.csv): This file contains the daily frequency price of BTC to USDT.
2. [full_df.csv](full_df.csv): This file contains the full dataset used for the project.

## Scripts

1. [preprocessing_ML.py](preprocessing_ML.py): This script is used for preprocessing the data for machine learning algorithms.
2. [ML_algo.ipynb](ML_algo.ipynb): This Jupyter notebook contains the machine learning algorithms used for predicting BTC returns.

## How to Run

To run this project, follow these steps:

1. Run the preprocessing script to prepare the data:

```sh
python preprocessing_ML.py

jupyter notebook ML_algo.ipynb
