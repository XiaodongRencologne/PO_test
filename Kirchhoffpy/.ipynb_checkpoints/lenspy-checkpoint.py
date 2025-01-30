
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

import pyvista as pv
pv.set_jupyter_backend('trame')#('static')#


# define the panel     
def squaresample(centerx,centery,sizex,sizey,Nx,Ny,surface,r0,r1,quadrature='uniform'):
    centerx=np.array(centerx);
    centery=np.array(centery);
    sizex=np.array(sizex);
    sizey=np.array(sizey);
    Nx=np.array(int(Nx));
    Ny=np.array(int(Ny))
        
    if quadrature.lower()=='uniform':
        x=np.linspace(-sizex/2+sizex/Nx/2,sizex/2-sizex/Nx/2,Nx);
        y=np.linspace(-sizey/2+sizey/Ny/2,sizey/2-sizey/Ny/2,Ny);
        xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2));
            
        x=xyarray[...,0];
        y=xyarray[...,1];
            
        M=Coord();
        M.x=(centerx.reshape(-1,1)+x).ravel();
        M.y=(centery.reshape(-1,1)+y).ravel();
        # surface can get the z value of the mirror and also can get the normal 
        M.z,Mn=surface(M.x,M.y);

        Masker = np.zeros(M.x.shape)
        NN = np.where(((M.x**2 + M.y**2)>=r0**2) & ((M.x**2 + M.y**2)<=r1**2))
        Masker[NN] = 1.0
        M.x = M.x * Masker
        M.y = M.y * Masker
        M.z = M.z * Masker
        Mn.x = Mn.x * Masker
        Mn.y = Mn.y * Masker
        Mn.z = Mn.z * Masker
        Mn.N = Mn.N * Masker
        dA=sizex*sizey/Nx/Ny;
        w = dA*Mn.N
    elif quadrature.lower()=='gaussian':
        print(1);
    return M,Mn,w;

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
        self.widget=pv.Plotter(notebook=True)
        _ = self.widget.add_axes(
            line_width=5,
            cone_radius=0.6,
            shaft_length=0.7,
            tip_length=0.3,
            ambient=0.5,
            label_size=(0.4, 0.16),
        )
        _ = self.widget.add_bounding_box(line_width=5, color='black')

    def analysis(self,N1,N2):
        if self.method == None:
            print('Please define the analysis methods!!!')
        else:
            data = self.method()
    def sampling(self,f1_N, f2_N, Sampling_type = 'polar',phi_type = 'uniform'):
        '''
        sampling_type = 'Gaussian' / 'uniform'
        '''
        f1 = Coord()
        f2 = Coord()
        def r1(theta):
            return np.ones(theta.shape)*self.diameter/2
        def r2(theta):
            return np.ones(theta.shape)*self.diameter/2
        if Sampling_type == 'polar':
            f1.x, f1.y1, f1.w= Guass_L_quadrs_Circ(0,r1,
                                        f1_N[0],f1_N[1],
                                        0,2*np.pi,f1_N[2],
                                        Phi_type=phi_type)
            f2.x, f2.y1, f2.w  = Guass_L_quadrs_Circ(0,r2,
                                        f2_N[0],f2_N[1],
                                        0,2*np.pi,f2_N[2],
                                        Phi_type=phi_type)
        elif Sampling_type = 'rectangle':
            f1.x, f1.y1, f1.w = Gauss_L_quadrs2d(-self.diameter/2,self.diameter/2,1,f1_N[0],
                                          -self.diameter/2,self.diameter/2,1,f1_N[1])
            f2.x, f2.y1, f2.w = Gauss_L_quadrs2d(-self.diameter/2,self.diameter/2,1,f2_N[0],
                                          -self.diameter/2,self.diameter/2,1,f2_N[1])
        
        f1.z,f1_n = self.surf_fnc1(f1.x, f1.y1)
        f1.z,f2_n= self.surf_fnc2(f2.x, f2.y1)

        return f1,f2,f1_n,f2_n
        """
        f1,f1_n,w1 = squaresample(0,0,self.diameter,self.diameter,
                                   f1_N[0],f1_N[1],self.surf_fnc1,
                                   0,self.diameter/2)
        f2,f2_n,w2 = squaresample(0,0,self.diameter,self.diameter,
                                   f1_N[0],f1_N[1],self.surf_fnc1,
                                   0,self.diameter/2)
        return f1,f2,f1_n,f2_n,w1,w2
        """
    def view(self,N1 = 101,N2 =101):
        if self.name+'_face1' in self.widget.actoers.keys():
            self.widget.remove_actor(self.widget.actors[self.name+'_face1'])
        x1 = np.linspace(0,self.diameter/2,N1)
        x2 = np.linspace(0,self.diameter/2,N2)
        z1 = self.surf_fnc1(r1,0)
        z2 = self.surf_fnc2(r2,0)
        p1 = pv.PolyData(np.column_stack((x1,np.zeros(x1.shape),z1)))
        p2 = pv.PolyData(np.column_stack((x2,np.zeros(x2.shape),z2)))
        p1 = p1.delaunay_2d()
        p2 = p2.delaunay_2d()

        view_face1 = p1.extrude_rotate(resolution=60)
        view_face2 = p2.extrude_rotate(resolution=60)

        self.widget.add_mesh(view_face1, color= 'lightblue' ,opacity= 0.8,name = self.name+'_face1')
        self.widget.add_mesh(view_face2, color= 'lightblue' ,opacity= 0.8,name = self.name+'_face2')
        self.widget.show()
        