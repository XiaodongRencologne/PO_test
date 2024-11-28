#%%
import numpy as np;
import torch as T;
import copy ;
import time;

# %%
from POpyGPU import PO_GPU as PO
from POpyGPU import PO_far
# 1. define the guassian beam of the input feed;
from Kirchhoffpy.Feedpy import Gaussibeam;
# 2. translation between coordinates system;
from Kirchhoffpy.coordinate_operations import Coord;
from Kirchhoffpy.coordinate_operations import Transform_local2global as local2global;
from Kirchhoffpy.coordinate_operations import Transform_global2local as global2local;
from Kirchhoffpy.coordinate_operations import cartesian_to_spherical as cart2spher;
# %%
from Kirchhoffpy.lenspy import simple_Lens