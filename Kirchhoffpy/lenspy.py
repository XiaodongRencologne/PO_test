#%%
"""
Package used to build mirror model based on 'uniform sampling' and 'Gaussian quadrature';

"""

import numpy as np;
import torch as T;
from .coordinate_operations import Coord;

from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;

#%%
def read_surface(file):
    pass


class simple_Lens():
    def __init__(self,
                 n,thickness, D,
                 surfac1, surface2,):
        self.n = n

