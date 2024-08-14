
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.load_cfg import load_yaml

from src.models.simple_transformer.transformer import SimpleTransformer
from src.models.simple_transformer.train import Trainer
from src.models.simple_transformer.dataset import create_dataloader, split_data


def load_data(cfg):
    filepath = cfg["training_path"]
    df = pd.read_csv(filepath, index_col=0)
    df = df.sort_values("Date", ascending=True)
    return df


def prepare_train_val_loaders(df, window_size, batch_size, device):
    train_df, val_df = split_data(df)

    train_loader, columns = create_dataloader(train_df, window_size, batch_size, device)
    val_loader, _ = create_dataloader(val_df, window_size, batch_size, device, columns=columns)

    return train_loader, val_loader, columns


def calculate_pos_weight(train_df):
    n_neg = len(train_df[train_df["Upward"] == 0])
    n_pos = len(train_df[train_df["Upward"] == 1])

    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float)
    print(f"pos_weight: {pos_weight}")
    return pos_weight


def initialize_model(cfg, device):
    embed_dim = cfg.get("embed_dim", 128)
    num_heads = cfg.get("num_heads", 8)
    num_layers = cfg.get("num_layers", 4)
    output_dim = cfg.get("output_dim", 1)
    num_time_varying_vars = cfg.get("num_time_varying_vars", 36)

    model = SimpleTransformer(embed_dim, num_heads, num_layers, output_dim, num_time_varying_vars)
    model.to(device)
    return model


def main():
    cfg = load_yaml("cfg.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data(cfg)

    window_size = cfg.get("window_size", 3)
    batch_size = cfg.get("batch_size", 128)
    train_loader, val_loader, _ = prepare_train_val_loaders(df, window_size, batch_size, device)

    pos_weight = calculate_pos_weight(df)
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = initialize_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 0.0005))

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, save_path=cfg["model_save_path"])
    trainer.train(cfg.get("epochs", 300))


if __name__ == "__main__":
    main()