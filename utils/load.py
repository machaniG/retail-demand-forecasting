"""load training data from provided data source, e.g. csv, database, etc."""
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path):
    """Load training data from file."""
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path} with shape {df.shape}")
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            logger.info(f"Loaded data from {data_path} with shape {df.shape}")
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
            logger.info(f"Loaded data from {data_path} with shape {df.shape}")
        elif data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
            logger.info(f"Loaded data from {data_path} with shape {df.shape}")
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return pd.DataFrame()


# load score data from csv file
def load_score_data(score_data_path):
    """Load score data from file."""
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