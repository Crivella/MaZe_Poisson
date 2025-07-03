from typing import Dict, Tuple

from .base_file_input import GridSetting, MDVariables, OutputSettings


def initialize_from_dict(data: Dict) -> Tuple[GridSetting, OutputSettings, MDVariables]:
    settings_map = [
        ('grid_setting', GridSetting),
        ('output_settings', OutputSettings),
        ('md_variables', MDVariables)
    ]

    res = []

    for key, cls in settings_map:
        args = cls.normalize_args(data[key])
        res.append(cls(**args))

    return tuple(res)