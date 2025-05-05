import logging
import os
import time

from . import MAIN_LOGGER_NAME

# Ensure logs are written to the script's directory
log_file_path = os.path.join(os.getcwd(), "all_logs.log")

class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        local_time = time.localtime(record.created)
        utc_offset = time.strftime('%z', local_time)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        return f"{formatted_time} {utc_offset}"

def add_file_handler(logger: logging.Logger, level: int = logging.DEBUG):
    # Create a file handler
    file_handler = logging.FileHandler(log_file_path, mode='a')  # Append mode
    file_formatter = UTCFormatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

def add_stream_handler(logger: logging.Logger, level: int = logging.INFO):
    # Add a stream handler to print to console
    stream_formatter = logging.Formatter('{message}', style='{')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(level)  # Ensure debug messages are not printed to the console
    logger.addHandler(stream_handler)

# Instantiate the main logger
logger = logging.getLogger(MAIN_LOGGER_NAME)
logger.setLevel(logging.DEBUG)  # Set the logging level
add_file_handler(logger)
add_stream_handler(logger)

def set_log_level(level: int):
    """Set the log level for the main logger and all its children."""
    for handler in logger.handlers:
        handler.setLevel(level)

# Function to set up the logger
def setup_logger(name, level: int = None):
    res = logger.getChild(name)
    
    if level is not None:
        res.setLevel(level)
    
    return res

class Logger:
    def __init__(self, *args, **kwargs):
        self.logger = setup_logger(self.__class__.__name__)
        super().__init__(*args, **kwargs)
    def __getattr__(self, name):
        return super().__getattribute__(name)
