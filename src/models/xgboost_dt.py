import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.load_cfg import load_yaml


class ModelTrainer:
    def __init__(
        self, filepath, model_type="decision_tree", target="Upward", train_size=0.7
    ):
        self.data = pd.read_csv(filepath)
        self.data = self.data.sort_values("Date")

        self.model_type = model_type
        self.target_col = target
        self.train_size = train_size
        self.prepare_data()
        self.train_test_split()
        self.model = self.initialize_model()

    def prepare_data(self):
        # No additional feature preparation; use all features as is.
        self.features = self.data.drop(columns=[self.target_col, "Date"])
        self.target = self.data[self.target_col]

    def train_test_split(self):
        split_index = int(len(self.data) * self.train_size)
        self.X_train = self.features[:split_index]
        self.y_train = self.target[:split_index]
        self.X_test = self.features[split_index:]
        self.y_test = self.target[split_index:]

        split_date = self.data.iloc[split_index]["Date"]
        print(f"Split index: {split_index}, Split date: {split_date}")

    def initialize_model(self):
        if self.model_type == "decision_tree":
            # Automatically balance the class weights
            return DecisionTreeClassifier(random_state=42, class_weight="balanced")
        elif self.model_type == "xgboost":
            num_pos = np.sum(self.y_train == 1)
            num_neg = np.sum(self.y_train == 0)
            scale_pos_weight = num_neg / num_pos

            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
            )
        elif self.model_type == "random_forest":
            # Automatically balance the class weights
            return RandomForestClassifier(random_state=42, class_weight="balanced")
        else:
            raise ValueError(
                "Invalid model type specified. Choose 'decision_tree', 'xgboost', or 'random_forest'."
            )

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        cm = confusion_matrix(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{report}")

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")


if __name__ == "__main__":
    cfg = load_yaml("cfg.yaml")

    filepath = cfg["training_path"]

    # Run Decision Tree
    dt_model = ModelTrainer(
        filepath, model_type="decision_tree", target="Upward", train_size=0.8
    )
    dt_model.train_model()
    print("Decision Tree Results")
    dt_model.evaluate_model()

    # Run XGBoost
    xgb_model = ModelTrainer(
        filepath, model_type="xgboost", target="Upward", train_size=0.8
    )
    xgb_model.train_model()
    print("Xgboost Results")
    xgb_model.evaluate_model()

    # Run Random Forest
    rf_model = ModelTrainer(
        filepath, model_type="random_forest", target="Upward", train_size=0.8
    )
    rf_model.train_model()
    print("Random Forest Results")
    rf_model.evaluate_model()
