import logging
import time

from . import MAIN_LOGGER_NAME


class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        local_time = time.localtime(record.created)
        utc_offset = time.strftime('%z', local_time)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        return f"{formatted_time} {utc_offset}"

# Instantiate the main logger
logger = logging.getLogger(MAIN_LOGGER_NAME)
logger.setLevel(logging.DEBUG)  # Set the logging level

stream_formatter = logging.Formatter('{message}', style='{')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(stream_formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

class Logger:
    def __init__(self, *args, log_level: int = logging.INFO, **kwargs):
        self.setup()
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return super().__getattribute__(name)

    def setup(self):
        """Set up the logger with the specified level."""
        main_logger = logging.getLogger(MAIN_LOGGER_NAME)

        child_logger = main_logger.getChild(self.__class__.__name__)
        # Inherit the level from the main logger to allow for disabling all loggers at once
        child_logger.setLevel(0)

        self.logger = child_logger

    def add_file_handler(self, path: str, level: int = logging.DEBUG):
        """Add a file handler to the logger."""
        file_handler = logging.FileHandler(path, mode='a')  # Append mode
        file_formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    def set_log_level(self, level: int):
        """Set the log level for the logger."""
        stream_handler.setLevel(level)
