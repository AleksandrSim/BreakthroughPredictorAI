import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import argrelextrema

from src.utils.load_cfg import load_yaml


class TargetCreator:
    def __init__(
        self,
        df: pd.DataFrame,
        timeframe="1m",
        ticker="BTC",
    ):
        self.df = df
        self.timeframe = timeframe
        self.ticker = ticker
        self.prepare_data()

    def prepare_data(self):
        self.df.set_index(["Date"], inplace=True)
        self.df.sort_index(ascending=True, inplace=True)
        self.df["close"] = self.df["Close"]
        # print(f'prepare_data took {time.time() - start_time:.4f} seconds')

    @staticmethod
    def get_extrema(data: np.array, order=5, k=2, pattern="higher_lows"):
        start_time = time.time()
        data = np.array(data)

        if pattern == "higher_lows":
            extrema_idx = argrelextrema(data, np.less, order=order)[0]
            compare = lambda curr, prev: curr > prev
        elif pattern == "lower_highs":
            extrema_idx = argrelextrema(data, np.greater, order=order)[0]
            compare = lambda curr, prev: curr < prev
        elif pattern == "higher_highs":
            extrema_idx = argrelextrema(data, np.greater, order=order)[0]
            compare = lambda curr, prev: curr > prev
        elif pattern == "lower_lows":
            extrema_idx = argrelextrema(data, np.less, order=order)[0]
            compare = lambda curr, prev: curr < prev
        else:
            raise ValueError(
                "Invalid pattern specified. Choose from \
                'higher_lows', 'lower_highs', 'higher_highs', 'lower_lows'."
            )

        extrema_values = data[extrema_idx]
        extrema = []
        ex_deque = deque(maxlen=k)

        for i, idx in enumerate(extrema_idx):
            if i == 0:
                ex_deque.append(idx)
                continue
            if not compare(extrema_values[i], extrema_values[i - 1]):
                ex_deque.clear()
            ex_deque.append(idx)
            if len(ex_deque) == k:
                extrema.append(ex_deque.copy())

        # print(f'get_extrema took {time.time() - start_time:.4f} seconds')
        return extrema

    def visualize_extrema(self, div: list, tr=None):
        if tr is None:
            tr = range(len(self.df))
        close = self.df["Close"].values[tr]
        dates = self.df.index[tr]

        hh, hl, ll, lh = div

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(15, 8))
        plt.plot(self.df["close"][tr])
        _ = [plt.plot(dates[i], close[i], c=colors[1]) for i in hh]
        _ = [plt.plot(dates[i], close[i], c=colors[2]) for i in hl]
        _ = [plt.plot(dates[i], close[i], c=colors[3]) for i in ll]
        _ = [plt.plot(dates[i], close[i], c=colors[4]) for i in lh]
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.title("Potential Divergence Points for BTC Closing Price")
        legend_elements = [
            Line2D([0], [0], color=colors[0], label="Close"),
            Line2D([0], [0], color=colors[1], label="Higher Highs"),
            Line2D([0], [0], color=colors[2], label="Higher Lows"),
            Line2D([0], [0], color=colors[3], label="Lower Lows"),
            Line2D([0], [0], color=colors[4], label="Lower Highs"),
        ]
        plt.legend(handles=legend_elements)
        plt.show()
        # print(f'visualize_extrema took {time.time() - start_time:.4f} seconds')

    def add_trend_columns(self, trend_up, trend_down):
        self.df["Upward"] = 0
        self.df["Downward"] = 0

        self.df["Upward_end"] = 0
        self.df["Downward_end"] = 0

        for up in trend_up:
            self.df.at[self.df.index[up[0]], "Upward"] = 1

        for down in trend_down:
            self.df.at[self.df.index[down[0]], "Downward"] = 1

    def connect_ranges(self, date_ranges):
        connected_ranges = []
        current_start, current_end = date_ranges[0]
        for rnge in date_ranges[1:]:
            start, end = rnge
            if current_end >= start:
                current_end = end
            else:
                connected_ranges.append(deque([current_start, current_end]))
                current_start, current_end = start, end
        connected_ranges.append(deque([current_start, current_end]))
        # print(f'connect_ranges took {time.time() - start_time:.4f} seconds')
        return connected_ranges

    def subtract_deque(self, main_range, subranges):
        result = []
        # Convert subranges to a list of tuples for sorting
        subranges = [tuple(subrange) for subrange in subranges]
        subranges = sorted(subranges, key=lambda x: x[0])
        current_start = main_range[0]
        current_end = main_range[1]
        for sub_start, sub_end in subranges:
            if current_start < sub_start - 1:
                result.append([current_start + 1, sub_start - 1])
            current_start = max(current_start + 1, sub_end)
        if current_start <= current_end:
            result.append([current_start + 1, current_end])
        # print(f'subtract_deque took {time.time() - start_time:.4f} seconds')
        return np.array(result)

    def calculate_trends(self, div: list, order: int, k: int):
        hh, hl, ll, lh = div

        close = self.df["close"].values

        hh = self.get_extrema(close, order, k, pattern="higher_highs")
        hl = self.get_extrema(close, order, k, pattern="higher_lows")
        ll = self.get_extrema(close, order, k, pattern="lower_lows")
        lh = self.get_extrema(close, order, k, pattern="lower_highs")

        hh_start_end = np.array([[ext[0], ext[-1]] for ext in hh])
        hl_start_end = np.array([[ext[0], ext[-1]] for ext in hl])
        ll_start_end = np.array([[ext[0], ext[-1]] for ext in ll])
        lh_start_end = np.array([[ext[0], ext[-1]] for ext in lh])

        def find_conflicts(start_end1, start_end2):
            overlaps = []
            for start1, end1 in start_end1:
                mask = (start_end2[:, 0] <= end1) & (start_end2[:, 1] >= start1)
                if np.any(mask):
                    overlap = np.column_stack(
                        (
                            np.maximum(start1, start_end2[mask, 0]),
                            np.minimum(end1, start_end2[mask, 1]),
                        )
                    )
                    overlaps.extend(overlap)
            return np.array(overlaps)

        up_conf = find_conflicts(hh_start_end, hl_start_end)
        down_conf = find_conflicts(ll_start_end, lh_start_end)

        date_range = (0, len(self.df.index) - 1)
        excluded_ranges = np.vstack((down_conf, up_conf))
        no_trend = self.subtract_deque(date_range, excluded_ranges)
        # print(f'calculate_trends took {time.time() - start_time:.4f} seconds')
        return up_conf, down_conf, no_trend

    def create_targets(self, order, k):
        hh = self.get_extrema(self.df.Close, order, k, pattern="higher_highs")
        hl = self.get_extrema(self.df.Close, order, k, pattern="higher_lows")
        ll = self.get_extrema(self.df.Close, order, k, pattern="lower_lows")
        lh = self.get_extrema(self.df.Close, order, k, pattern="lower_highs")

        up_conf, down_conf, no_trend = self.calculate_trends(
            div=[hh, hl, ll, lh], order=order, k=k
        )

        # print(f'create_targets took {time.time() - start_time:.4f} seconds')
        return up_conf, down_conf, no_trend

    def visualize_trends(self, trend_up, trend_down, no_trend, tr=None):
        if tr is None:
            tr = range(len(self.df))
        close = self.df["close"].values[tr]
        dates = self.df.index[tr]

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(15, 8))
        plt.plot(self.df["close"][tr])
        _ = [plt.plot(dates[i], close[i], c=colors[3]) for i in trend_down]
        _ = [plt.plot(dates[i], close[i], c=colors[2]) for i in trend_up]
        _ = [plt.plot(dates[i], close[i], c=colors[1]) for i in no_trend]
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.title(f"Trends for {self.ticker} Closing Price")
        legend_elements = [
            Line2D([0], [0], color=colors[0], label="Close"),
            Line2D([0], [0], color=colors[3], label="Trend Down"),
            Line2D([0], [0], color=colors[2], label="Trend Up"),
            Line2D([0], [0], color=colors[1], label="No Trend"),
        ]
        plt.legend(handles=legend_elements)
        plt.show()
        # print(f'visualize_trends took {time.time() - start_time:.4f} seconds')


if __name__ == "__main__":
    cfg = load_yaml("cfg.yaml")
    K = cfg["K"]
    ORDER = cfg["Order"]

    data = pd.read_csv(cfg["btc_path"])
    from src.prepare.prepare_features import Preprocessor

    preprocessor = Preprocessor(
        data,
        indicators=["SMA", "EMA", "RSI", "BB"],
        parent_range=["2020-01-01", "2020-01-05"],
    )

    preprocessor.preprocess(date_range=["2020-01-02", "2024-01-03"])

    start = time.time()
    df = preprocessor.data

    target_creator = TargetCreator(df)
    hh = target_creator.get_extrema(df.Close, ORDER, K, pattern="higher_highs")
    hl = target_creator.get_extrema(df.Close, ORDER, K, pattern="higher_lows")
    ll = target_creator.get_extrema(df.Close, ORDER, K, pattern="lower_lows")
    lh = target_creator.get_extrema(df.Close, ORDER, K, pattern="lower_highs")

    trend_up, trend_down, no_trend = target_creator.create_targets(ORDER, K)
    target_creator.visualize_trends(trend_up, trend_down, no_trend)
    print(f"end {time.time() - start}")
