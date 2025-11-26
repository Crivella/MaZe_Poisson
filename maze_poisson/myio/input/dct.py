from typing import Dict, Tuple

from .base_file_input import GridSetting, MDVariables, OutputSettings


def initialize_from_dict(data: Dict) -> Tuple[GridSetting, OutputSettings, MDVariables]:
    settings_map = [
        ('grid_setting', GridSetting),
        ('output_settings', OutputSettings),
        ('md_variables', MDVariables)
    ]

    res = [cls.from_dict(data[key]) for key, cls in settings_map]

    return tuple(res)