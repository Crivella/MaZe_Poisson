"""Handle global IO"""
enabled: bool = True

def disable():
    global enabled
    enabled = False

def get_enabled() -> bool:
    return enabled

from .loggers import Logger, logger
from .output import OutputFiles
from .progress_bar import ProgressBar

__all__ = [
    'Logger', 'logger', 'OutputFiles', 'ProgressBar'
]
