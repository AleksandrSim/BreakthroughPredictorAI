import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.models.simple_transformer.dataset import create_dataloader, split_data
from src.models.simple_transformer.transformer import SimpleTransformer
from src.utils.load_cfg import load_yaml


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, device, save_path
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        # Create save path directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            sequences, targets = batch
            targets = targets.squeeze()  #
            outputs = self.model(sequences).squeeze()
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():  #
            for batch in tqdm(self.val_loader, desc="Validation"):
                sequences, targets = batch
                targets = targets.squeeze()
                outputs = self.model(sequences).squeeze()  # Model outputs

                # Apply sigmoid activation and round to get binary predictions
                preds = torch.round(torch.sigmoid(outputs))

                # Convert predictions and targets to NumPy arrays and store them
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Check for NaN values in predictions (just in case)
        if np.isnan(all_preds).any():
            print("Warning: NaN values found in predictions.")
            all_preds = np.nan_to_num(
                all_preds
            )  # Replace NaNs with 0 or another appropriate value

        # Calculate the confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        return cm

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch()
            #            cm = self.validate()
            print(f"Loss: {train_loss}")
            print("Confusion Matrix:")
            #            print(cm)
            # Save the model state
            torch.save(
                self.model.state_dict(), f"{self.save_path}/model_epoch_{epoch + 1}.pth"
            )
            print(f"Model saved to {self.save_path}/model_epoch_{epoch + 1}.pth")


def load_model(model, model_path):
    state_dict = torch.load(model_path)

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

    # If you want to move the model to GPU (if available)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


if __name__ == "__main__":
    cfg = load_yaml("cfg.yaml")

    filepath = cfg["training_path"]
    df = pd.read_csv(filepath, index_col=0)
    df = df.sort_values("Date", ascending=True)

    train_df, val_df = split_data(df)

    n_neg = len(train_df[train_df["Upward"] == 0])
    n_pos = len(train_df[train_df["Upward"] == 1])

    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float)

    print(f"pos_weight: {pos_weight}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    window_size = 3
    batch_size = 128

    train_loader, columns = create_dataloader(train_df, window_size, batch_size, device)
    val_loader, _ = create_dataloader(
        val_df, window_size, batch_size, device, columns=columns
    )

    for sequences, targets in val_loader:
        print(f"shape:{sequences.shape},shape: {targets.shape}")
        break  # Just pr

    embed_dim = 128
    num_heads = 8
    num_layers = 4
    output_dim = 1
    num_time_varying_vars = 36
    epochs = 300

    model = SimpleTransformer(
        embed_dim, num_heads, num_layers, output_dim, num_time_varying_vars
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        save_path=cfg["model_save_path"],
    )
    trainer.train(epochs)
