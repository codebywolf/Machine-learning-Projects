import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'configs/roberta_config.yaml'):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration file successfully loaded from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Configuration file could not be loaded: {e}")
        raise e