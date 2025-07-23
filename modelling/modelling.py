import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_train_data(train_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(train_path)
        logging.info(f"Training data loaded from {train_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def extract_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X = df.drop(columns=['label']).values
        y = df['label'].values
        logging.info("Features and labels extracted from training data")
        return X, y
    except Exception as e:
        logging.error(f"Failed to extract features/labels: {e}")
        raise

def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        logging.info("RandomForest model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        n_estimators = params['model_training']['n_estimators']
        max_depth = params['model_training']['max_depth']
        train_data = load_train_data("data/interim/train_bow.csv")
        X_train, y_train = extract_features_labels(train_data)
        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()