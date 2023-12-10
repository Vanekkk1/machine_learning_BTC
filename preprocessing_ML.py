import ccxt
import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def fetch_crypto_data(symbol: str, timeframe: str, start_date: str) -> pd.DataFrame:
    exchange = ccxt.binance()
    limit = 1000
    since = exchange.parse8601(f"{start_date}T00:00:00Z")

    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if len(ohlcv) == 0:
                break

            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 60 * 1000
        except ccxt.NetworkError as e:
            print(f"Network error: {e}")
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}")
            break
        except ccxt.BaseError as e:
            print(f"CCXT base error: {e}")
            break

    symbol_modified = symbol.replace("/", ":")
    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            f"{symbol_modified}_timestamp",
            f"{symbol_modified}_open",
            f"{symbol_modified}_high",
            f"{symbol_modified}_low",
            f"{symbol_modified}_close",
            f"{symbol_modified}_volume",
        ],
    )

    df[f"{symbol_modified}_timestamp"] = pd.to_datetime(
        df[f"{symbol_modified}_timestamp"], unit="ms"
    )
    df.set_index(f"{symbol_modified}_timestamp", inplace=True)

    file_name = f"{symbol.replace('/', ':')}_price_{timeframe}freq.csv"
    df.to_csv(file_name)

    return df

def update_crypto_data(symbol: str, timeframe: str):
    # Modify the symbol for the file name
    modified_symbol = symbol.replace('/', ':')

    file_name = f"{modified_symbol}_price_{timeframe}freq.csv"

    # Load existing DataFrame
    try:
        existing_df = pd.read_csv(file_name, index_col=0)
        existing_df.index = pd.to_datetime(existing_df.index)
    except FileNotFoundError:
        print(f"No existing data found for {symbol} with {timeframe} timeframe. Fetching new data.")
        existing_df = pd.DataFrame()

    # Determine the last timestamp in the existing DataFrame
    if not existing_df.empty:
        last_timestamp = existing_df.index[-1]
        new_data_start_date = (last_timestamp + pd.Timedelta(hours=1) if timeframe == "1h" else last_timestamp + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        new_data_start_date = "2017-08-17"  # Default start date

    # Fetch new data starting from the last timestamp
    # Ensure the symbol format matches what fetch_crypto_data expects
    new_df = fetch_crypto_data(symbol, timeframe, new_data_start_date)

    # Concatenate the old and new data
    updated_df = pd.concat([existing_df, new_df]).drop_duplicates()

    # Save the updated DataFrame back to CSV
    updated_df.to_csv(file_name)

    return updated_df

def add_regression_target(
    df: pd.DataFrame, symbol: str, steps_to_forecast: int, timeframe: str
) -> pd.DataFrame:
    symbol = symbol.replace("/", ":")
    shift = steps_to_forecast * (24 if timeframe == "1h" else 1)
    df[f"{symbol}_target"] = (
        df[f"{symbol}_close"].pct_change(periods=shift).shift(-shift)
    )
    return df


def add_class_target(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    steps_to_forecast: int,
    class_threshold=0.03,
) -> pd.DataFrame:
    symbol = symbol.replace("/", ":")
    shift = steps_to_forecast * (24 if timeframe == "1h" else 1)
    df[f"{symbol}_target"] = (
        df[f"{symbol}_close"].pct_change(periods=shift).shift(-shift)
    )
    # Applying the classification logic based on the threshold
    df.loc[
        df[f"{symbol}_target"] > class_threshold, f"{symbol}_target"
    ] = 2  # Buy position
    df.loc[
        df[f"{symbol}_target"] < -class_threshold, f"{symbol}_target"
    ] = 1  # Short position
    df.loc[
        (df[f"{symbol}_target"] <= class_threshold)
        & (df[f"{symbol}_target"] >= -class_threshold),
        f"{symbol}_target",
    ] = 0  # Hold
    return df


def get_features_and_target(
    symbol: str,
    feature_lags: List[int] = [3, 9, 16],
    steps_to_forecast: int = 1,
    class_threshold=0.03,
    model_type="reg",
    model_freq="1h",
) -> pd.DataFrame:
    symbol = symbol.replace("/", ":")

    features_df = pd.read_csv(
        f"{symbol}_price_{model_freq}freq.csv",
        parse_dates=True,
        index_col=f"{symbol}_timestamp",
    )
    required_columns = [
        f"{symbol}_close",
        f"{symbol}_high",
        f"{symbol}_low",
        f"{symbol}_volume",
    ]
    if not all(col in features_df.columns for col in required_columns):
        raise ValueError("Required columns are missing in the DataFrame")

    # Moving Averages
    for ma in [9, 20, 50]:
        features_df[f"{symbol}_sma_{ma}"] = talib.SMA(
            features_df[f"{symbol}_close"], timeperiod=ma
        )

    # RSI
    features_df[f"{symbol}_rsi"] = talib.RSI(
        features_df[f"{symbol}_close"], timeperiod=14
    )

    # Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(
        features_df[f"{symbol}_close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    features_df[f"{symbol}_bollinger_up"] = upperband
    features_df[f"{symbol}_bollinger_down"] = lowerband

    # ADX
    features_df[f"{symbol}_adx"] = talib.ADX(
        features_df[f"{symbol}_high"],
        features_df[f"{symbol}_low"],
        features_df[f"{symbol}_close"],
        timeperiod=14,
    )

    # MACD
    macd, macdsignal, macdhist = talib.MACD(
        features_df[f"{symbol}_close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    features_df[f"{symbol}_macd_diff"] = macd - macdsignal

    # OBV
    features_df[f"{symbol}_obv"] = talib.OBV(
        features_df[f"{symbol}_close"], features_df[f"{symbol}_volume"]
    )

    # Ichimoku Cloud
    nine_period_high = features_df[f"{symbol}_high"].rolling(window=9).max()
    nine_period_low = features_df[f"{symbol}_low"].rolling(window=9).min()
    features_df[f"{symbol}_ichimoku_conversion"] = (
        nine_period_high + nine_period_low
    ) / 2

    # Stochastic Oscillator
    stochastic_k, stochastic_d = talib.STOCH(
        features_df[f"{symbol}_high"],
        features_df[f"{symbol}_low"],
        features_df[f"{symbol}_close"],
    )
    features_df[f"{symbol}_stochastic_k"] = stochastic_k
    features_df[f"{symbol}_stochastic_d"] = stochastic_d

    # Aroon Indicator
    aroon_up, aroon_down = talib.AROON(
        features_df[f"{symbol}_high"], features_df[f"{symbol}_low"], timeperiod=14
    )
    features_df[f"{symbol}_aroon_up"] = aroon_up
    features_df[f"{symbol}_aroon_down"] = aroon_down

    # Lagged metrics and deltas calculation
    for lag in feature_lags:
        features_df[f"{symbol}_rsi_lag_{lag}"] = features_df[f"{symbol}_rsi"].shift(lag)
        features_df[f"{symbol}_macd_diff_lag_{lag}"] = features_df[
            f"{symbol}_macd_diff"
        ].shift(lag)
        features_df[f"{symbol}_obv_lag_{lag}"] = features_df[f"{symbol}_obv"].shift(lag)
        features_df[f"{symbol}_ichimoku_conversion_lag_{lag}"] = features_df[
            f"{symbol}_ichimoku_conversion"
        ].shift(lag)
        features_df[f"{symbol}_stochastic_k_lag_{lag}"] = stochastic_k.shift(lag)
        features_df[f"{symbol}_stochastic_d_lag_{lag}"] = stochastic_d.shift(lag)
        features_df[f"{symbol}_aroon_up_lag_{lag}"] = aroon_up.shift(lag)
        features_df[f"{symbol}_aroon_down_lag_{lag}"] = aroon_down.shift(lag)

        features_df[f"{symbol}_rsi_delta_{lag}"] = features_df[f"{symbol}_rsi"].diff(
            lag
        )
        features_df[f"{symbol}_macd_diff_delta_{lag}"] = features_df[
            f"{symbol}_macd_diff"
        ].diff(lag)
        features_df[f"{symbol}_obv_delta_{lag}"] = features_df[f"{symbol}_obv"].diff(
            lag
        )
        features_df[f"{symbol}_ichimoku_conversion_delta_{lag}"] = features_df[
            f"{symbol}_ichimoku_conversion"
        ].diff(lag)
        features_df[f"{symbol}_stochastic_k_delta_{lag}"] = stochastic_k.diff(lag)
        features_df[f"{symbol}_stochastic_d_delta_{lag}"] = stochastic_d.diff(lag)
        features_df[f"{symbol}_aroon_up_delta_{lag}"] = aroon_up.diff(lag)
        features_df[f"{symbol}_aroon_down_delta_{lag}"] = aroon_down.diff(lag)

    if model_type == "reg":
        features_df = add_regression_target(
            df=features_df,
            symbol=symbol,
            steps_to_forecast=steps_to_forecast,
            timeframe=model_freq,
        )
    elif model_type == "class":
        features_df = add_class_target(
            df=features_df,
            timeframe=model_freq,
            symbol=symbol,
            steps_to_forecast=steps_to_forecast,
            class_threshold=class_threshold,
        )
    else:
        raise ValueError("not supported model type")

    features_df.drop(
        columns=[
            f"{symbol}_open",
            f"{symbol}_high",
            f"{symbol}_low",
            f"{symbol}_close",
            f"{symbol}_volume",
        ],
        inplace=True,
    )
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.dropna(inplace=True)

    return features_df


def get_ML_dfs(
    symbol: str,
    feature_lags: List[int] = [3, 9, 16],
    steps_to_forecast: int = 7,
    random_state: int = 99,
    fetch_data_params: Optional[Dict[str, any]] = None,
    testing_hours: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares data for machine learning by generating features and splitting into train and test sets.

    :param symbol: The symbol for the cryptocurrency pair (e.g., 'BTC/USDT').
    :param feature_lags: List of integers representing the lags for feature generation.
    :param steps_to_forecast: Number of days ahead to forecast.
    :param random_state: An integer seed for random number generator for reproducible splits. Used only when 'testing_hours' is None.
    :param fetch_data_params: Optional dictionary of parameters to pass to the fetch_crypto_data function. If provided, new data is fetched using these parameters.
    :param testing_hours: Optional integer specifying the number of latest data points to use for the test set. If provided, splits the data into training and test sets based on this value. If None, performs a standard train-test split.
    :return: Tuple of (X_train, X_test, y_train, y_test), where 'X' contains the features and 'y' is the target variable.
    """
    symbol_modified = symbol.replace("/", ":")

    # Fetch new data if fetch_data_params is provided
    if fetch_data_params is not None:
        fetch_crypto_data(symbol, **fetch_data_params)

    df = get_features_and_target(symbol_modified, feature_lags, steps_to_forecast)
    X = df.drop(columns=f"{symbol_modified}_target")
    y = df[f"{symbol_modified}_target"].copy()

    if testing_hours:
        X_train, X_test, y_train, y_test = (
            X[:-testing_hours],
            X[-testing_hours:],
            y[:-testing_hours],
            y[-testing_hours:],
        )

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_state
        )

    return (X_train, X_test, y_train, y_test)
