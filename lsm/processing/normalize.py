import os
import numpy as np
from typing import Optional


def normalize_image(
    y: np.ndarray,
    lower: Optional[int] = 1,
    upper: Optional[int] = 99,
) -> np.ndarray:
    """normalize the image to [0, 1] by making use of the upper
    lower intensity percentiles"""
    x = y.copy()
    # get the lower and upper percentiles of data
    x_pc_lower = np.percentile(x, lower)
    x_pc_upper = np.percentile(x, upper)
    x = (x - x_pc_lower) / (x_pc_upper - x_pc_lower)

    return x
