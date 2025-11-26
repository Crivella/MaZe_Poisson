import json
from typing import Dict, Tuple

from .base_file_input import (GridSetting, MDVariables, OutputSettings,
                              mpi_file_loader)
from .dct import initialize_from_dict


@mpi_file_loader
def read_json_file(filename: str) -> Dict:
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def initialize_from_json(filename: str) -> Tuple[GridSetting, OutputSettings, MDVariables]:
    data = read_json_file(filename)

    return initialize_from_dict(data)
