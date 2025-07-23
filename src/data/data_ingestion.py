import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

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

def fetch_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logging.info(f"Data fetched from {url}")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=['tweet_id'])
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Data preprocessing completed")
        return final_df
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train and test sets with test_size={test_size}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Data splitting failed: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, train_path: str, test_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train and test data saved to {train_path} and {test_path}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        test_size = float(params['data_ingestion']['test_size'])
        url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        df = fetch_data(url)
        final_df = preprocess_data(df)
        train_data, test_data = split_data(final_df, test_size)
        save_data(train_data, test_data, "data/raw/train.csv", "data/raw/test.csv")
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()