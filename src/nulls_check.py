import pandas as pd

from utils.load_cfg import load_yaml


def convert_timestamp_to_date(df, timestamp_col="Timestamp"):
    df["Timestamp"] = pd.to_datetime(df[timestamp_col], unit="s")
    return df


def check_null_values(df):
    null_summary = df.isnull().sum()
    return null_summary[null_summary > 0]


def count_null_and_non_null_timestamps(df):
    null_rows = df[df.isnull().any(axis=1)]
    non_null_rows = df.dropna()
    return len(null_rows), len(non_null_rows)


def locate_nulls(df):
    return df[df.isnull().any(axis=1)]


def describe_data(df):
    return df.describe(), df.info()


def analyze_nulls_by_year(df, date_col="Timestamp"):
    df["Year"] = df[date_col].dt.year
    nulls_by_year = df[df.isnull().any(axis=1)].groupby("Year").size()
    total_by_year = df.groupby("Year").size()
    percentage_nulls_by_year = (nulls_by_year / total_by_year) * 100
    nulls_analysis = pd.DataFrame(
        {
            "Total Nulls": nulls_by_year,
            "Total Entries": total_by_year,
            "Percentage Nulls": percentage_nulls_by_year,
        }
    ).fillna(
        0
    )  # Fill NaN with 0 for years with no nulls

    return nulls_analysis


if __name__ == "__main__":
    cfg = load_yaml("cfg.yaml")
    df = pd.read_csv(cfg["btc_path"])

    # Convert timestamp to standard date
    df = convert_timestamp_to_date(df)

    # Check null values
    null_summary = check_null_values(df)
    print("Null Value Summary:")
    print(null_summary)

    # Count timestamps with and without null values
    total_null_count, total_non_null_count = count_null_and_non_null_timestamps(df)
    print(f"\nTotal Timestamps with Null Values: {total_null_count}")
    print(f"Total Timestamps without Null Values: {total_non_null_count}")

    nulls_by_year = analyze_nulls_by_year(df)
    print("\nNull Analysis by Year:")
    print(nulls_by_year)
