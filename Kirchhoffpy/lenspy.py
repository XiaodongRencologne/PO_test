
#%%
"""
Package used to build mirror model based on 'uniform sampling' and 'Gaussian quadrature';

"""

import numpy as np;
import torch as T;
from scipy.interpolate import CubicSpline

from .Gauss_L_quadr import Guass_L_quadrs_Circ
from .coordinate_operations import Coord;
from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;

#%%
def read_rsf(file):
    data = np.genfromtxt(file, skip_header = 2)
    r = data[:,0]
    z = data[:,1]
    cs = CubicSpline(r,z)
    def srf(x,y):
        r = np.sqrt(x**2+y**2)
        z = cs(r)
        n = Coord()
        # surface normal vector
        n.z = -np.ones(x.shape)
        n.x = cs(r,1)*x/r
        n.y = cs(r,1)*y/r
        n.N= np.sqrt(n.x**2+n.y**2+1)
        n.x=n.x/n.N
        n.y=n.y/n.N
        n.z=n.z/n.N
        return z, n
    return srf



class simple_Lens():
    def __init__(self,
                 n,thickness, D,
                 surface_file1, surface_file2,
                 name = 'simplelens'):
        self.name = name
        self.n = n
        self.t = thickness
        self.diameter = D
        self.surf_fnc1 = read_rsf(surface_file1)
        self.surf_fnc2 = read_rsf(surface_file2)

        # define surface for sampling or for 3Dview
        self.f1 = Coord()
        self.f1_n = Coord()
        self.f2 = Coord()
        self.f2_n = Coord()
        # 3D view
        # Analysis method
        self.method = None

    def analysis(self,N1,N2):
        if self.method == None:
            print('Please define the analysis methods!!!')
        else:
            data = self.method()
    def sampling(self,f1_N, f2_N,):
        '''
        sampling_type = 'Gaussian' / 'uniform'
        '''
        def r1(theta):
            return np.ones(theta.shape)*self.diameter/2
        def r2(theta):
            return np.ones(theta.shape)*self.diameter/2
        x1,y1,w1 = Guass_L_quadrs_Circ(0,r1,
                                        f1_N[0],f1_N[1],
                                        0,2*np.pi,f1_N[2],
                                        Phi_type='uniform')
        x2,y2,w2 = Guass_L_quadrs_Circ(0,r2,
                                    f2_N[0],f2_N[1],
                                    0,2*np.pi,f2_N[2],
                                    Phi_type='uniform')
        z1 = self.surf_fnc1(x1,y1)
        z2 = self.surf_fnc2(x2,y2)
        return x1,y1,z1,w1,x2,y2,z2,w2
    def view_sampling(self,f1_N=[21,121],f2_N=[21,121]):
        N1 = [f1_N[0],0,f1_N[1]]
        N2 = [f2_N[0],0,f2_N[1]]
        self.view_1 = Coord()
        self.view_2 = Coord()
        self.view_1.x,self.view_1.y,self.view_1.z,self.view_1.w,\
            self.view_2.x,self.view_2.y,self.view_1.z,self.view_2.w = self.sampling(N1,N2)
    def view(self):
        pass




#%%
