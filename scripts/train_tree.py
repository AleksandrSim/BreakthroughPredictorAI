from src.models.xgboost_dt import ModelTrainer
from src.utils.load_cfg import load_yaml


def run_model(filepath, model_type, target, train_size):
    """Train and evaluate the model specified by model_type."""
    model = ModelTrainer(
        filepath, model_type=model_type, target=target, train_size=train_size
    )
    model.train_model()
    print(f"\n{model_type.capitalize()} Results")
    model.evaluate_model()
    return model


def main():
    """Main function to run Decision Tree and XGBoost models."""
    # Load configuration
    cfg = load_yaml("cfg.yaml")

    # Extract filepath and other configurations
    filepath = cfg.get("training_path", "")
    target = cfg.get("target_column", "Upward")
    train_size = cfg.get("train_size", 0.8)

    # Run Decision Tree Model
    run_model(
        filepath, model_type="decision_tree", target=target, train_size=train_size
    )

    # Run XGBoost Model
    run_model(filepath, model_type="xgboost", target=target, train_size=train_size)


if __name__ == "__main__":
    main()
