"""Handle global IO"""
import logging

enabled: bool = True
MAIN_LOGGER_NAME = "main"

def disable_io():
    global enabled
    enabled = False
    logging.getLogger(MAIN_LOGGER_NAME).setLevel(logging.CRITICAL)

def get_enabled() -> bool:
    return enabled

from .loggers import Logger, logger
from .output import OutputFiles
from .progress_bar import ProgressBar

__all__ = [
    'Logger', 'logger', 'OutputFiles', 'ProgressBar'
]
