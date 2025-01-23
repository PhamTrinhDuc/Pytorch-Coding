from dataclasses import dataclass

@dataclass
class BaseServeArgs:
    config_name: str = "raw_data"
    model_name: str = "retnest18"
    model_alias: str = "production"
    