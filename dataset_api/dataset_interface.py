# dataset_api/dataset_interface.py
from abc import ABC, abstractmethod
from enum import Enum

class DatasetFunctions(Enum):
    GET_LOADER = "get_loader"
    EVALUATE = "evaluate"
    COLLATE = "collate"
    WRAPPER = "evaluator_wrapper"
    CONFIG = "get_config"

class DatasetInterface(ABC):
    def get_function(self, name: str):
        """Return a dataset-specific callable (e.g., 'evaluate', 'get_loader')"""
        raise NotImplementedError(f"Function '{name}' not implemented in this dataset interface.")

    def get_config(self) -> dict:
        """Return the dataset's default configuration dictionary"""
        raise NotImplementedError("get_config() not implemented in this dataset interface.")
    
    def get_config_value(self, key: str, default=None):
        config = self.get_config()
        if key in config:
            return config[key]
        if default is not None:
            return default
        raise KeyError(f"[CONFIG ERROR] Key '{key}' not found in config and no default provided.")