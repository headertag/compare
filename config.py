import torch
import yaml

def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_device(device_str):
    """Determines the processing device (CPU or CUDA)."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Load the configuration
config = load_config()

# -- Configurable Parameters --
# Telegram
TELEGRAM_TOKEN = config["telegram"]["token"]
TELEGRAM_CHAT_IDS = config["telegram"]["chat_ids"]

# Alerting
ALERT_COOLDOWN = config["alerting"]["cooldown_seconds"]
ALERT_COOLDOWN_THRESHOLD = config["alerting"]["cooldown_threshold_multiplier"]
MIN_ALERT_INTERVAL = config["alerting"]["min_interval_seconds"]
ALERT_SENSITIVITY_THRESHOLD = config["alerting"]["sensitivity_threshold"]

# Camera
CAM_INDEX = config["camera"]["index"]
CAM_WIDTH = config["camera"]["width"]
CAM_HEIGHT = config["camera"]["height"]

# Processing
DEVICE = get_device(config["processing"]["device"])
COCO_CLASSES_PATH = config["processing"]["coco_classes_path"]

# Models
MODELS_CONFIG = config["models"]

# Dashboard
REMOVE_BACKGROUND = config["dashboard"]["remove_background"]
