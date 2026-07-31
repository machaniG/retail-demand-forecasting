"""save trained model to disk .pkl file with joblib
and save model metadata to .json file"""

import json
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
MODEL_PATH = "models/"
FEATURES_PATH = "models/"


def save_model(model, model_name="lgbm_model.pkl"):
    """Save the trained model to disk as a .pkl file using joblib."""
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_save_path = os.path.join(MODEL_PATH, model_name)
    joblib.dump(model, model_save_path)
    logger.info(f"Model saved to {model_save_path}")



def save_model_metadata(model, metadata, metadata_name="model_metadata.json"):
    """Save model metadata to disk as a .json file."""
    os.makedirs(MODEL_PATH, exist_ok=True)
    metadata_save_path = os.path.join(MODEL_PATH, metadata_name)
    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f)
    logger.info(f"Model metadata saved to {metadata_save_path}")


# save feature list to disk as a .json file
def save_feature_list(feature_list, feature_list_name="feature_list.json"):
    """Save feature list to disk as a .json file."""
    os.makedirs(FEATURES_PATH, exist_ok=True)
    feature_list_save_path = os.path.join(FEATURES_PATH, feature_list_name)
    with open(feature_list_save_path, "w") as f:
        json.dump(feature_list, f)
    logger.info(f"Feature list saved to {feature_list_save_path}")