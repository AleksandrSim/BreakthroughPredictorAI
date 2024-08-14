import pandas as pd

from src.prepare.prepare_target import TargetCreator
from src.utils.load_cfg import load_yaml


class Preprocessor:
    INTERVALS = {"1m": 1, "5m": 5, "15m": 15}

    def __init__(self, data, indicators, filter=False, order=5, k=2):
        self.data = data
        self.data.columns = self.data.columns.str.title()
        self.filter = filter
        self.order = order
        self.k = k
        self.indicators = indicators

    def convert_timestamp(self):
        self.data["Date"] = pd.to_datetime(
            self.data["Timestamp"], unit="s"
        ).dt.tz_localize(None)
        self.data.drop("Timestamp", axis=1, inplace=True)

    def filter_data(self, start_date, end_date=None):
        if end_date:
            self.data = self.data[
                (self.data["Date"] >= pd.to_datetime(start_date))
                & (self.data["Date"] <= pd.to_datetime(end_date))
            ]
        else:
            self.data = self.data[self.data["Date"] > pd.to_datetime(start_date)]

    def label_sustained_movements(self):
        trend_identifier = TargetCreator(self.data[["Close", "Date"]])

        trend_up, trend_down, no_trend = trend_identifier.create_targets(
            self.order, self.k
        )
        trend_identifier.add_trend_columns(trend_up, trend_down)
        if self.filter:
            trend_identifier.filter_trends()

        trend_identifier.df.drop(["Close", "close"], axis=1, inplace=True)
        trend_identifier.df = trend_identifier.df.reset_index()

        self.data = pd.merge(self.data, trend_identifier.df, on="Date", how="left")

    def add_moving_averages_rsi_and_bollinger_bands(self):
        new_data = {}
        for label, minutes in self.INTERVALS.items():
            if "SMA" in self.indicators:
                new_data[f"SMA_{label}"] = (
                    self.data["Close"].rolling(window=minutes).mean()
                )
            if "EMA" in self.indicators:
                new_data[f"EMA_{label}"] = (
                    self.data["Close"].ewm(span=minutes, adjust=False).mean()
                )
            if "RSI" in self.indicators:
                delta = self.data["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=minutes).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=minutes).mean()
                rs = gain / loss
                new_data[f"RSI_{label}"] = 100 - (100 / (1 + rs))
            if "BB" in self.indicators:
                sma = self.data["Close"].rolling(window=minutes).mean()
                std = self.data["Close"].rolling(window=minutes).std()
                new_data[f"BB_upper_{label}"] = sma + (std * 2)
                new_data[f"BB_lower_{label}"] = sma - (std * 2)
                new_data[f"BB_middle_{label}"] = sma

        new_data_df = pd.DataFrame(new_data)
        self.data = pd.concat([self.data, new_data_df], axis=1)

    def add_fibonacci_retracement(self):
        new_data = {}
        for label, minutes in self.INTERVALS.items():
            window_high = self.data["High"].rolling(window=minutes).max()
            window_low = self.data["Low"].rolling(window=minutes).min()
            range_ = window_high - window_low

            new_data[f"Fib_23.6_{label}"] = window_low + 0.236 * range_
            new_data[f"Fib_38.2_{label}"] = window_low + 0.382 * range_
            new_data[f"Fib_50_{label}"] = window_low + 0.5 * range_
            new_data[f"Fib_61.8_{label}"] = window_low + 0.618 * range_

        new_data_df = pd.DataFrame(new_data)
        self.data = pd.concat([self.data, new_data_df], axis=1)

    def preprocess(self, date_range):
        self.convert_timestamp()
        print("Timestamp conversion is done")

        self.filter_data(date_range[0], date_range[1])
        print("Data filtering is done")

        self.add_moving_averages_rsi_and_bollinger_bands()
        print("Adding moving averages, RSI, and Bollinger Bands is done")

        self.add_fibonacci_retracement()
        print("Adding Fibonacci retracement levels is done")

        self.label_sustained_movements()
        print("Labeling sustained movements is done")


if __name__ == "__main__":
    import time

    start = time.time()

    cfg = load_yaml("cfg.yaml")
    data = pd.read_csv(cfg["btc_path"])
    data = data.dropna()
    print(data.head)

    preprocessor = Preprocessor(
        data, indicators=["SMA", "EMA", "RSI", "BB"], filter=False
    )
    preprocessor.preprocess(date_range=["2019-01-01", "2021-02-10"])
    print(preprocessor.data.head())
    preprocessor.data.to_csv(cfg["training_path"])

    print(preprocessor.data.head(10))

    print(f"Time taken: {time.time() - start} seconds")
