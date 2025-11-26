import os
from typing import Tuple

from .base_file_input import GridSetting, MDVariables, OutputSettings
from .json import initialize_from_json
from .yaml import initialize_from_yaml

__all__ = ['load_file']

initializer_map = {
    '.yaml': initialize_from_yaml,
    '.yml': initialize_from_yaml,
    '.json': initialize_from_json,
}

def load_file(file_path: str) -> Tuple[GridSetting, OutputSettings, MDVariables]:
    """Get the initializer for the file based on the extension."""
    ext = os.path.splitext(file_path)[1]

    if ext not in initializer_map:
        raise ValueError(f'File extension {ext} not supported')

    return initializer_map[ext](file_path)
