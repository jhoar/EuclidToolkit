import numpy as np
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import math

# Local
_ROOT: Path = Path('db')

# Euclid SOC workspace in DL
# _ROOT: Path = Path.home() / 'team_workspaces' / 'Euclid-SOC' / 'FGS' / 'db'

RAD_TO_MAS:float = 3600.0 * 1000.0 * 180.0 / math.pi

def ndarray_to_list(array: np.ndarray) -> list:
    """
    Converts numpy array into list preserving empty dimensions (e.g. it will not produce a scalar out of a single element)
    """
    if len(array.shape) == 1:
        return [element.item() for element in array]
    else:
        return [ndarray_to_list(element) for element in array]
    
@dataclass
class TimeRange:
    start: datetime
    end: datetime
