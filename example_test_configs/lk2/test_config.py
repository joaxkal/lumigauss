import numpy as np

config = {
        "01-08_07_30_IMG_6710.JPG": {
            "mask_path": "/path/to/mask/01-08_07_30_IMG_6710.png",
            "env_map_path": "/path/to/envmap/ENV_MAP_CC/01-08_07_30/20210801_082411.jpg",
            "initial_env_map_rotation": {
                "x": np.pi * (-90 / 180),
                "y": 0,
                "z": 0,
            },
            "env_map_scaling": {
                "threshold": 0.999,
                "scale": 10,
            },
            "sun_angles": [-0.1 * np.pi, 0.4 * np.pi],
        },
        "08-08_16_00_IMG_7850.JPG": {
            "mask_path": "/path/to/mask/08-08_16_00_IMG_7850.png",
            "env_map_path": "/path/to/envmap/ENV_MAP_CC/08-08_16_00/20210808_164134.jpg",
            "initial_env_map_rotation": {
                "x": np.pi * (-90 / 180),
                "y": 0,
                "z": 0,
            },
            "env_map_scaling": {
                "threshold": 0.999,
                "scale": 10,
            },
            "sun_angles": [0.6 * np.pi, 1.0 * np.pi],
        },
        "28-07_10_00_DSC_0055.jpg": {
            "mask_path": "/path/to/mask/28-07_10_00_DSC_0055.png",
            "env_map_path": "/path/to/envmap/ENV_MAP_CC/28-07_10_00/20210728_122650.jpg",
            "initial_env_map_rotation": {
                "x": np.pi * (-90 / 180),
                "y": 0,
                "z": 0,
            },
            "env_map_scaling": {
                "threshold": 0.999,
                "scale": 10,
            },
            "sun_angles": [0.8 * np.pi, 1.1 * np.pi],
        },
        "29-07_12_00_IMG_5424.JPG": {
            "mask_path": "/path/to/mask/29-07_12_00_IMG_5424.png",
            "env_map_path": "/path/to/envmap/ENV_MAP_CC/29-07_12_00/20210729_125318.jpg",
            "initial_env_map_rotation": {
                "x": np.pi * (-90 / 180),
                "y": 0,
                "z": 0,
            },
            "env_map_scaling": {
                "threshold": 0.999,
                "scale": 10,
            },
            "sun_angles": [-0.3 * np.pi, 0.2 * np.pi],
        },
        "29-07_20_30_IMG_5607.JPG": {
            "mask_path": "/path/to/mask/29-07_20_30_IMG_5607.png",
            "env_map_path": "/path/to/envmap/ENV_MAP_CC/29-07_20_30/20210729_203644.jpg",
            "initial_env_map_rotation": {
                "x": np.pi * (-90 / 180),
                "y": 0,
                "z": 0,
            },
            "env_map_scaling": {
                "threshold": 0.999,
                "scale": 10,
            },
            "sun_angles": [0.1 * np.pi, 0.6 * np.pi],
        },
}
