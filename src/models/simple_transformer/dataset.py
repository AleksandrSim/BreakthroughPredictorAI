import numpy as np
import pandas as pd
import torch


def prepare_sequences(data, targets, window_size):
    sequences = []
    sequence_targets = []
    for i in range(len(data) - window_size):
        seq_data = data[i : i + window_size].clone()
        if i > 0:
            previous_targets = targets[i : i + window_size - 1].squeeze()
            seq_data[:, -1] = torch.cat(
                (previous_targets, torch.tensor([0.0], dtype=torch.float32))
            ) 
        sequences.append(seq_data)
        sequence_targets.append(targets[i + window_size])
    return torch.stack(sequences), torch.stack(sequence_targets)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_dataloader(df, window_size, batch_size, device, columns=None):
    print(f"Initial DataFrame length: {len(df)}")

    # If a list of columns is provided, select those columns from the DataFrame
    if columns:
        df = df[columns]  # Ensure 'Date' and 'Upward' columns are included

    columns_to_drop = []
    for col in df.columns:
        if col == "Date":
            # Convert Date to Unix timestamp (milliseconds)
            df[col] = (
                pd.to_datetime(df[col]).astype("int64") // 10**9
            )  # seconds since epoch
        else:
            try:
                df[col] = pd.to_numeric(df[col], downcast="integer", errors="raise")
            except ValueError:
                columns_to_drop.append(col)

    if columns_to_drop:
        print(f"Columns dropped (could not convert to integer): {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    df = df.reset_index(drop=True)  # Reset index after dropping rows

    # Check for NaN or infinite values
    if df.isnull().values.any():
        print("Warning: DataFrame contains NaN values. Consider cleaning your data.")
        # Display columns with NaN values and their counts
        nan_columns = df.columns[df.isnull().any()]
        print("Columns with NaN values:")
        for col in nan_columns:
            print(f"{col}: {df[col].isnull().sum()} NaN values")

        # Drop columns with more than 100 NaN values
        nan_threshold = 100
        columns_to_drop = [
            col for col in nan_columns if df[col].isnull().sum() > nan_threshold
        ]
        if columns_to_drop:
            print(
                f"Dropping columns with more than {nan_threshold} NaN values: {columns_to_drop}"
            )
            df.drop(columns=columns_to_drop, inplace=True)

        # Drop rows with remaining NaN values
        df.dropna(inplace=True)
        print(f"DataFrame length after dropping NaN rows: {len(df)}")

    if np.isinf(df.values).any():
        print(
            "Warning: DataFrame contains infinite values. Consider cleaning your data."
        )
        # Display columns with infinite values and their counts
        inf_columns = df.columns[np.isinf(df).any()]
        print("Columns with infinite values:")
        for col in inf_columns:
            print(f"{col}: {np.isinf(df[col]).sum()} infinite values")

        # Replace infinite values with NaN and drop rows with NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        print(f"DataFrame length after dropping NaN/Inf rows: {len(df)}")

    df = df.reset_index(drop=True)  # Reset index after dropping rows

    # Prepare data for the DataLoader
    data = torch.tensor(df.drop(columns=["Upward"]).values, dtype=torch.float32)
    targets = torch.tensor(df["Upward"].values, dtype=torch.float32).unsqueeze(1)

    sequences, sequence_targets = prepare_sequences(data, targets, window_size)
    dataset = CustomDataset(sequences, sequence_targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    final_columns = (
        df.columns.tolist()
    )  # Get the final list of columns after processing

    return dataloader, final_columns


def split_data(df):
    train_data = df[df["Date"] < "2020-09-11"]
    val_data = df[df["Date"] >= "2020-09-11"]
    return train_data, val_data
