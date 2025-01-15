import os

from .base_file_input import GridSetting, MDVariables, OutputSettings
from .yaml import initialize_from_yaml

__all__ = ['load_file']

initializer_map = {
    '.yaml': initialize_from_yaml,
    '.yml': initialize_from_yaml,
}

def load_file(file_path: str) -> tuple[GridSetting, OutputSettings, MDVariables]:
    """Get the initializer for the file based on the extension."""
    ext = os.path.splitext(file_path)[1]

    if ext not in initializer_map:
        raise ValueError(f'File extension {ext} not supported')

    return initializer_map[ext](file_path)
