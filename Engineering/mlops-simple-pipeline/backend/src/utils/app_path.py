from pathlib import Path


class AppPath:
    ROOT_DIR = Path(__file__).parent.parent.parent # cd to backend folder
    
    SOURCE_DIR = ROOT_DIR / "src" # cd to src folder

    CONFIG_DIR = SOURCE_DIR / "config"
    SERVE_CONFIG = CONFIG_DIR / "serve_config"
    DATA_CONFIG = CONFIG_DIR / "data_config"

    DATA_DIR = ROOT_DIR / "raw_data"
    RAW_DATA_DIR = DATA_DIR / "catdog_raw"
    COLLECTED_DATA_DIR = DATA_DIR / "collected"
    TRAIN_DATA_DIR = DATA_DIR / "train_data"

AppPath.COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)


class LoggingConfig:
    ROOT_DIR = Path(__file__).parent.parent.parent # cd to backend folder

    SOURCE_DIR = ROOT_DIR / "src" # cd to src folder
    LOG_DIR = SOURCE_DIR / "logs"

LoggingConfig.LOG_DIR.mkdir(parents=True, exist_ok=True) # create logs folder in src 