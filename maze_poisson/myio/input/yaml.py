from typing import Dict, Tuple

import yaml

from .base_file_input import (GridSetting, MDVariables, OutputSettings,
                              mpi_file_loader)
from .dct import initialize_from_dict


@mpi_file_loader
def read_yaml_file(filename: str) -> Dict:
    """Read a YAML file and return the initialized objects."""
    with open(filename, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    return data

def initialize_from_yaml(filename: str) -> Tuple[GridSetting, OutputSettings, MDVariables]:
    data = read_yaml_file(filename)

    return initialize_from_dict(data)
