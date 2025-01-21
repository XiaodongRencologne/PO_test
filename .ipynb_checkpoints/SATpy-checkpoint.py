#%%
import numpy as np;
import torch as T;
import copy ;
import time;

# %%
from POpyGPU import PO_GPU,PO_far_GPU, lensPO, lensPO_far
# 1. define the guassian beam of the input feed;
from Kirchhoffpy.Feedpy import Gaussibeam
from Kirchhoffpy.lenspy import simple_Lens
# 2. translation between coordinates system;
from Kirchhoffpy.coordinate_operations import Coord;
from Kirchhoffpy.coordinate_operations import Transform_local2global as local2global;
from Kirchhoffpy.coordinate_operations import Transform_global2local as global2local;
from Kirchhoffpy.coordinate_operations import cartesian_to_spherical as cart2spher;
# %%
from Kirchhoffpy.lenspy import simple_Lens
SILICON = 3.36
L_lensFp_3   = 7.177590111674096
L_lens3_2    = 15.586806616226909
L_lens2_1    = 57.632802785493645
L_lens1_Lyot = 1.162050628144469
L_Ly_vw      = 22.7114

L_lens1_ref = L_lensFp_3 + L_lens3_2 + L_lens2_1
L_lens2_ref = L_lensFp_3 + L_lens3_2
L_lens3_ref = L_lensFp_3 
L_Ly_ref = L_lens1_ref + L_lens1_Lyot
L_vw_ref = L_Ly_ref + L_Ly_vw

class SATpo():
    def __init__(self, 
                 freq='90GHz',
                 Rx_position=[0,0,0],
                 feed_Beam={'Gaussian':{'T_angle_x':0,
                                        'E_tape_x': 0,
                                        'T_angle_y':0,
                                        'E_tape_x':0}}):
            self.Rx = Rx_position
    
    def _coord(self):
        self.coor_L1 = {'angle': [],
                        'D': []}
        self.coor_L2 = {'angle': [],
                        'D': []}
        self.coor_feed ={'angle':[np.pi,0,0],
                         'D':self.Rx}
        self.coor_cut = {'angle':[np.pi,0,0],
                         'D':[0,0,0]}
    def _Geo(self):
        self.L1 = simple_lens()
