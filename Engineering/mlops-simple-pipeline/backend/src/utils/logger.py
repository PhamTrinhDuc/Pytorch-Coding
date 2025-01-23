import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
from logging.handlers import RotatingFileHandler
from app_path import LoggingConfig


class Logger:
    def __init__(self,
                 name="", 
                 log_level: str=logging.INFO,
                 log_file=None):
        self.log = logging.getLogger(name=name)
        self.get_logger(log_level, log_file)

    def get_logger(self, log_level: str, log_file: str):
        self.log.setLevel(level=log_level)
        self._init_formatter()

        if log_file is not None: # if log_file exist
            self._add_file_handler(log_file=LoggingConfig.LOG_DIR/log_file)
        else: # print log to console
            self._add_stream_handler()
    
    def _init_formatter(self):
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def _add_stream_handler(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.log.addHandler(stream_handler)
    
    def _add_file_handler(self, log_file: Path):
        file_handler = RotatingFileHandler(filename=log_file, maxBytes=10000, backupCount=10)
        file_handler.setFormatter(self.formatter)
        self.log.addHandler(file_handler)

    def log_model(self, predictor_name: str):
        self.log.info(f"Predictor name: {predictor_name}")
    
    def log_response(self, pred_prob: float, pred_id: int, pred_class: str): 
        self.log.info(f"Predicted prob: {pred_prob} - Predicted id: {pred_id} - Predicted class: {pred_class}")

