import logging
import pandas as pd
import pickle
import json
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_model(model_path: str) -> Any:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(test_data_path)
        logging.info(f"Test data loaded from {test_data_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
    try:
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation completed")
        return metrics
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise

def main() -> None:
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")
        metrics = evaluate_model(model, test_data)
        save_metrics(metrics, "reports/metrics.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()