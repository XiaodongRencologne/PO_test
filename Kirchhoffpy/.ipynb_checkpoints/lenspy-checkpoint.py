
#%%
"""
Package used to build mirror model based on 'uniform sampling' and 'Gaussian quadrature';

"""

import numpy as np
import torch as T
import scipy
from scipy.interpolate import CubicSpline

from .Gauss_L_quadr import Guass_L_quadrs_Circ,Gauss_L_quadrs2d
from .coordinate_operations import Coord;
from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;

from .LensPO import Fresnel_coeffi,poyntingVector,Z0,lensPO,PO_GPU

from .Vopy import vector,abs_v,scalarproduct

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
def read_rsf(file,units= 'cm'):
    factor= 1.0
    if units == 'cm':
        factor = 10
    elif units == 'm':
        factor = 1000
    data = np.genfromtxt(file, skip_header = 2)
    r = data[:,0]*factor
    z = data[:,1]*factor
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
def read_rsf2(file,units= 'cm'):
    factor= 1.0
    if units == 'cm':
        factor = 10
    elif units == 'm':
        factor = 1000
    data = np.genfromtxt(file, skip_header = 2)
    r = data[:,0]**2*factor**2
    z = data[:,1]*factor
    cs = CubicSpline(r,z)
    factor= 1.0
    if units == 'cm':
        factor = 10
    elif units == 'm':
        factor = 1000
    def srf(x,y):
        r = x**2+y**2
        z = cs(r)
        n = Coord()
        # surface normal vector
        n.z = -np.ones(x.shape)
        n.x = cs(r,1)*2*x
        n.y = cs(r,1)*2*y
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
                 widget,
                 coord,
                 name = 'simplelens',
                 Device = T.device('cuda')):
        self.name = name
        self.n = n
        self.t = thickness
        self.diameter = D
        self.surf_fnc1 = read_rsf2(surface_file1,units= 'cm')
        self.surf_fnc2 = read_rsf2(surface_file2,units= 'cm')
        self.coord = coord

        # define surface for sampling or for 3Dview
        self.f1 = Coord()
        self.f1_n = Coord()
        self.f2 = Coord()
        self.f2_n = Coord()
        # 3D view
        # Analysis method
        self.method = None
        self.widget = widget

        ## coordinate system of the two surfaces of the lens.

        '''
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
        '''
    def analysis(self,N1,N2,feed,k,
                 sampling_type_f1='rectangle',
                 phi_type_f1 = 'uniform',
                 sampling_type_f2='rectangle',
                 phi_type_f2 = 'uniform',):
        if self.method == None:
            print('Please define the analysis methods!!!')
        else:
            self.f1,self.f1_n = self.sampling(N1,self.surf_fnc1,self.r1,
                                              Sampling_type = sampling_type_f1,
                                              phi_type=phi_type_f1)
            
            self.f1_n =scalarproduct(-1,self.f1_n)
            
            self.f2,self.f2_n = self.sampling(N2,self.surf_fnc2,self.r2,
                                             Sampling_type = sampling_type_f2,
                                             phi_type=phi_type_f2)
            
            self.f2 = local2global([np.pi,0,0], [0,0,self.t],self.f2)
            self.f2_n = local2global([np.pi,0,0],[0,0,0],self.f2_n)
            
            self.f_E_in,self.f_H_in, E_co, E_cx = feed(self.f1,self.f1_n)
            
            self.f2_E,self.f2_H, self.f2_E_t, self.f2_E_r, self.f2_H_t,\
            self.f2_H_r, self.f1_E_t, self.f1_E_r,  self.f1_H_t , self.f1_H_r = self.method(self.f1,self.f1_n,self.f1.w,
                                                   self.f2,self.f2_n,
                                                   self.f_E_in,self.f_H_in,
                                                   k,self.n,
                                                   device = T.device('cuda'))
            
    def sampling(self, f1_N, surf_fuc,r1,r0=0,Sampling_type = 'polar', phi_type = 'uniform'):
        '''
        sampling_type = 'Gaussian' / 'uniform'
        '''
        f1 = Coord()
        #f2 = Coord()

        if Sampling_type == 'polar':
            f1.x, f1.y, f1.w= Guass_L_quadrs_Circ(0,r1,
                                        f1_N[0],f1_N[1],
                                        0,2*np.pi,f1_N[2],
                                        Phi_type=phi_type)
        elif Sampling_type == 'rectangle':
            f1.x, f1.y, f1.w = Gauss_L_quadrs2d(-self.diameter/2,self.diameter/2,f1_N[0],f1_N[1],
                                          -self.diameter/2,self.diameter/2,f1_N[2],f1_N[3])
            NN = np.where((f1.x**2+f1.y**2)>(self.diameter/2)**2)
            Masker = np.ones(f1.x.shape)
            Masker[NN] =0.0
            f1.w = f1.w*Masker
            f1.masker = Masker
        
        f1.z,f1_n = surf_fuc(f1.x, f1.y)

        return f1,f1_n

    def view(self,N1 = 101,N2 =101):
        if self.name+'_face1' in self.widget.actors.keys():
            self.widget.remove_actor(self.widget.actors[self.name+'_face1'])
        self.v_x1 = np.linspace(0,self.diameter/2,N1,dtype = np.float64)
        self.v_x2 = np.linspace(0,self.diameter/2,N2,dtype = np.float64)
        self.v_z1, self.v_n1 = self.surf_fnc1(self.v_x1/10,0)
        self.v_z2, self.v_n2 = self.surf_fnc2(self.v_x2/10,0)
        
        self.v_z1 = self.v_z1*10 +self.coord[-1]
        self.v_z2 = -self.v_z2*10 +self.coord[-1] +self.t
        p1 = pv.PolyData(np.column_stack((self.v_x1,np.zeros(self.v_x1.shape,dtype = np.float64),
                                          self.v_z1)))
        p2 = pv.PolyData(np.column_stack((self.v_x2,np.zeros(self.v_x2.shape,dtype = np.float64),
                                          self.v_z2)))
        p1 = p1.delaunay_2d()
        p2 = p2.delaunay_2d()

        view_face1 = p1.extrude_rotate(resolution=100)
        view_face2 = p2.extrude_rotate(resolution=100)

        self.widget.add_mesh(view_face1, color= 'lightblue' ,opacity= 1,name = self.name+'_face1')
        self.widget.add_mesh(view_face2, color= 'lightblue' ,opacity= 1,name = self.name+'_face2')
        # check surface normal vector
        
        cent = np.column_stack((self.v_x1,np.zeros(self.v_x1.shape),self.v_z1))
        direction =  np.column_stack((self.v_n1.x,self.v_n1.y,self.v_n1.z))
        self.widget.add_arrows(cent,direction*10,mag =1)

        '''
        cent = np.column_stack((self.v_x2,np.zeros(self.v_x2.shape),self.v_z2))
        direction =  np.column_stack((self.v_n2.x,self.v_n2.y,self.v_n2.z))
        self.widget.add_arrows(cent,direction*10,mag =1)
        '''
        '''
        n1 = 1.0
        Z1 = Z0/n1
        n2 = 3
        Z2 =Z0/n2
        E = vector()
        E.x = np.zeros(self.v_x1.shape,dtype = np.float64)
        E.y = np.sqrt(Z1)*np.ones(self.v_x1.shape,dtype = np.float64)
        E.z = np.zeros(self.v_x1.shape,dtype = np.float64)
        E.totensor()

        H = vector()
        H.x = -np.sqrt(Z1)*np.ones(self.v_x1.shape,dtype = np.float64)/Z1
        H.y = np.zeros(self.v_x1.shape,dtype = np.float64)
        H.z = np.zeros(self.v_x1.shape,dtype = np.float64)
        H.totensor()
        
        k_v1 = poyntingVector(E,H)
        cent = np.column_stack((self.v_x1,np.zeros(self.v_x1.shape),self.v_z1))
        direction =  np.column_stack((k_v1.x,k_v1.y,k_v1.z))
        self.widget.add_arrows(cent,direction*10,mag =1)
        self.v_n1.np2Tensor()

        #print('input power:', abs_v(E)**2/Z1)
        E_t,E_r,H_t,H_r = Fresnel_coeffi(n1,n2,self.v_n1,E,H)
        
        #print('output power:', (abs_v(E_t)**2/Z2).numpy())
        #print('reflection power:', (abs_v(E_r)**2/Z1).numpy())
        k_v1 = poyntingVector(E_t,H_t)
        cent = np.column_stack((self.v_x1,np.zeros(self.v_x1.shape),self.v_z1))
        direction =  np.column_stack((k_v1.x,k_v1.y,k_v1.z))
        self.widget.add_arrows(cent,direction*10,mag =1)
        #self.widget.show()
        '''
    def view2(self,N1 = [1,11,1],N2 =[1,11,1]):
        if self.name+'_face1' in self.widget.actors.keys():
            self.widget.remove_actor(self.widget.actors[self.name+'_face1'])
        f1,f2,f1_n,f2_n = self.sampling(N1, N2, Sampling_type = 'polar')
        f1.z = f1.z + self.coord[-1]
        f2.z = f2.z + self.coord[-1]
        #print(self.name)
        #print(np.acos(f1_n.z)*180/np.pi)
        #print(np.acos(-f2_n.z)*180/np.pi)
        
        
        p1 = pv.PolyData(np.column_stack((f1.x,f1.y,f1.z)))
        p2 = pv.PolyData(np.column_stack((f2.x,f2.y,f2.z)))
        p1 = p1.delaunay_2d()
        p2 = p2.delaunay_2d()
        view_face1 = p1.extrude_rotate(resolution=100)
        view_face2 = p2.extrude_rotate(resolution=100)

        self.widget.add_mesh(view_face1, color= 'lightblue' ,opacity= 1,name = self.name+'_face1')
        self.widget.add_mesh(view_face2, color= 'lightblue' ,opacity= 1,name = self.name+'_face2')
        # check surface normal vector
        
        cent = np.column_stack((f1.x,f1.y,f1.z))
        direction =  np.column_stack((f1_n.x,f1_n.y,f1_n.z))
        self.widget.add_arrows(cent,direction*20,mag =0.5)
        
        cent = np.column_stack((f2.x,f2.y,f2.z))
        direction =  np.column_stack((-f2_n.x,-f2_n.y,-f2_n.z))
        self.widget.add_arrows(cent,direction*20,mag =0.5)

    def view3(self,N1 = [11,1,11,1],N2 =[11,1,11,1]):
        if self.name+'_face1' in self.widget.actors.keys():
            self.widget.remove_actor(self.widget.actors[self.name+'_face1'])
        if self.name+'_face2' in self.widget.actors.keys():
            self.widget.remove_actor(self.widget.actors[self.name+'_face2'])
        if self.name+'_n1' in self.widget.actors.keys():
            self.widget.remove_actor(self.widget.actors[self.name+'_n1'])
        if self.name+'_n2' in self.widget.actors.keys():
            self.widget.remove_actor(self.widget.actors[self.name+'_n2'])
        f1,f2,f1_n,f2_n = self.sampling(N1, N2, Sampling_type = 'rectangle')
        f1.z = f1.z + self.coord[-1]
        f2.z = f2.z + self.coord[-1]
        """
        print(self.name)
        print((np.asin(f1_n.y)*180/np.pi).reshape(N1[0]*N1[1],N1[2]*N1[3]))
        print((np.asin(-f2_n.y)*180/np.pi).reshape(N2[0]*N2[1],N2[2]*N2[3]))

        print((np.acos(f1_n.z)*180/np.pi).reshape(N1[0]*N1[1],N1[2]*N1[3]))
        print((np.acos(-f2_n.z)*180/np.pi).reshape(N2[0]*N2[1],N2[2]*N2[3]))
        """
        grid1 = pv.StructuredGrid()
        grid1.points = np.c_[f1.x.ravel(), f1.y.ravel(), f1.z.ravel()]
        grid1.dimensions = (N1[0]*N1[1],N1[2]*N1[3], 1)

        grid2 = pv.StructuredGrid()
        grid2.points = np.c_[f2.x.ravel(), f2.y.ravel(), f2.z.ravel()]
        grid2.dimensions = (N2[0]*N2[1],N2[2]*N2[3], 1)
        self.widget.add_mesh(grid1, color= 'lightblue' ,opacity= 0.5,name = self.name+'_face1',show_edges=True)
        self.widget.add_mesh(grid2, color= 'lightblue' ,opacity= 0.5,name = self.name+'_face2',show_edges=True)
        # check surface normal vector
        
        cent = np.column_stack((f1.x,f1.y,f1.z))
        direction =  np.column_stack((f1_n.x,f1_n.y,f1_n.z))
        self.widget.add_arrows(cent,direction*20,mag =1,name = self.name+'_n1')
        
        cent = np.column_stack((f2.x,f2.y,f2.z))
        direction =  np.column_stack((-f2_n.x,-f2_n.y,-f2_n.z))
        self.widget.add_arrows(cent,direction*20,mag =1,name = self.name+'_n2')
        
    def r1(self,theta):
        return np.ones(theta.shape)*self.diameter/2
    def r2(self,theta):
            return np.ones(theta.shape)*self.diameter/2
# %%
