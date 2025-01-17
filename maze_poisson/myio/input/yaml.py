from typing import Tuple

import yaml

from .base_file_input import (GridSetting, MDVariables, OutputSettings,
                              mpi_file_loader, validate_all)


@mpi_file_loader
def initialize_from_yaml(filename: str) -> Tuple[GridSetting, OutputSettings, MDVariables]:
    with open(filename, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    grid_setting = GridSetting()
    output_settings = OutputSettings()
    md_variables = MDVariables()
    settings_map = {
        'grid_setting': grid_setting,
        'output_settings': output_settings,
        'md_variables': md_variables
    }

    for key in settings_map.keys():
        ptr = settings_map[key]
        for k, v in data[key].items():
            setattr(ptr, k, v)

    validate_all(grid_setting, output_settings, md_variables)

    return grid_setting, output_settings, md_variables
