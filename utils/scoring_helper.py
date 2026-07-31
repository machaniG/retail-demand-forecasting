"""helper functions for scoring pipeline"""
import os
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
MODEL_PATH = "models/"
MODEL_NAME = "lgbm_model.pkl"


# load model from disk
def load_model(model_name=MODEL_NAME):
    """Load model from disk."""
    model_load_path = os.path.join(MODEL_PATH, model_name)
    try:
        model = joblib.load(model_load_path)
        logger.info(f"Model loaded from {model_load_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_load_path}: {e}")
        return None


# load score data from csv file
def load_score_data(score_data_path):
    """Load scoring data from file."""
    try:
        if score_data_path.endswith('.csv'):
            df = pd.read_csv(score_data_path)
            logger.info(f"Loaded score data from {score_data_path} with shape {df.shape}")
        elif score_data_path.endswith('.parquet'):
            df = pd.read_parquet(score_data_path)
            logger.info(f"Loaded score data from {score_data_path} with shape {df.shape}")
        elif score_data_path.endswith('.json'):
            df = pd.read_json(score_data_path)
            logger.info(f"Loaded score data from {score_data_path} with shape {df.shape}")
        elif score_data_path.endswith('.xlsx'):
            df = pd.read_excel(score_data_path)
            logger.info(f"Loaded score data from {score_data_path} with shape {df.shape}")
        else:
            raise ValueError(f"Unsupported file format: {score_data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading score data from {score_data_path}: {e}")
        return pd.DataFrame()


def load_feature_list(feature_list_name="feature_list.json"):
    """Load feature list from disk."""
    feature_list_path = os.path.join(MODEL_PATH, feature_list_name)
    try:
        import json
        with open(feature_list_path, "r") as f:
            feature_list = json.load(f)
        logger.info(f"Feature list loaded from {feature_list_path}")
        return feature_list
    except Exception as e:
        logger.error(f"Error loading feature list from {feature_list_path}: {e}")
        return []


def save_predictions(output_df, output_name="predictions.csv"):
    """Save predictions to disk."""
    output_path = os.path.join(MODEL_PATH, output_name)
    try:
        output_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving predictions to {output_path}: {e}")