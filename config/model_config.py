# config/model_config.py
import yaml
import os
from pydantic import BaseModel
from typing import Dict, Any

# Define the structure/blueprint of config
class ModelSettings(BaseModel):
    MODEL_TYPE: str
    LIGHTGBM_PARAMS: Dict[str, Any]

# Load the actual values dynamically from the YAML file
config_path = os.path.join(os.path.dirname(__file__), "base_config.yaml")
with open(config_path, "r") as f:
    raw_yaml = yaml.safe_load(f)

# Parse and validate the values against the blueprint
settings = ModelSettings(**raw_yaml)