import addpath
from src.preprocessing import Preprocessing,  load_file
import numpy as np

data = np.eye(10)
# data[np.diag_indices(3)] = np.arange(1, 4)

p = Preprocessing(data, dtype=float)
p.clean(components=[0], scale_in=False)
p.reset()

r = np.arange(5)
even = np.array([False, True] * 5)
i = np.arange(10)

p.filter((even, '==', True, 0), include_nan=False)
p.filter((i, '<', 5, 0))