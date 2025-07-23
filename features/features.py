import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

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

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info(f"Train and test data loaded from {train_path} and {test_path}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load train/test data: {e}")
        raise

def extract_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X = df['content'].values
        y = df['sentiment'].values
        logging.info("Features and labels extracted")
        return X, y
    except Exception as e:
        logging.error(f"Failed to extract features/labels: {e}")
        raise

def vectorize_data(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Data vectorization completed")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Vectorization failed: {e}")
        raise

def save_feature_data(X_bow, y, path: str) -> None:
    try:
        df = pd.DataFrame(X_bow.toarray())
        df['label'] = y
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Feature data saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save feature data: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        max_features = params['feature_engineering']['max_features']
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train, y_train = extract_features_labels(train_data)
        X_test, y_test = extract_features_labels(test_data)
        X_train_bow, X_test_bow, _ = vectorize_data(X_train, X_test, max_features)
        save_feature_data(X_train_bow, y_train, "data/interim/train_bow.csv")
        save_feature_data(X_test_bow, y_test, "data/interim/test_bow.csv")
        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()