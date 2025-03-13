
#%%
"""
Package used to build mirror model based on 'uniform sampling' and 'Gaussian quadrature';

"""

import numpy as np
import copy
import h5py
import torch as T
import scipy
from scipy.interpolate import CubicSpline

from .Gauss_L_quadr import Guass_L_quadrs_Circ,Gauss_L_quadrs2d
from .coordinate_operations import Coord;
from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;
from .field_storage import Spherical_grd

from .LensPO import Fresnel_coeffi,poyntingVector,Z0,lensPO
from .POpyGPU import PO_far_GPU,PO_far,PO_GPU,epsilon,mu

from .Vopy import vector,abs_v,scalarproduct, CO

import pyvista as pv
pv.set_jupyter_backend('trame')#('static')#

import matplotlib.pyplot as plt
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

## save the lens surface information into H5 files, includeing surface, normal, field E and H
def saveh5_surf(file_h5,face,face_n, E, H, T, R,name = 'face'):
    group = file_h5.create_group(name)
    group.create_dataset('x', data = face.x)
    group.create_dataset('y', data = face.y)
    group.create_dataset('z', data = face.z)
    group.create_dataset('w', data = face.w)
    group.create_dataset('nx', data = face_n.x)
    group.create_dataset('ny', data = face_n.y)
    group.create_dataset('nz', data = face_n.z)
    group.create_dataset('N', data = face_n.N)
    group.create_dataset('Ex', data = E.x)
    group.create_dataset('Ey', data = E.y)
    group.create_dataset('Ez', data = E.z)
    group.create_dataset('Hx', data = H.x)
    group.create_dataset('Hy', data = H.y)
    group.create_dataset('Hz', data = H.z)
    group.create_dataset('T', data = T)
    group.create_dataset('R', data = R)
    #group.create_dataset('poynting', data = poynting)
## save the lens surface information into H5 files, includeing surface, normal, field E and H

def read_cur(filename):
    face = Coord()
    face_n = Coord()
    H = vector()
    with h5py.File(filename,'r') as f:
        face.x = f['f2/x'][:].ravel()
        face.y = f['f2/y'][:].ravel()
        face.z = f['f2/z'][:].ravel()
        face.w = f['f2/w'][:].ravel()
        face_n.x = f['f2/nx'][:].ravel()
        face_n.y = f['f2/ny'][:].ravel()
        face_n.z = f['f2/nz'][:].ravel()
        face_n.N = f['f2/N'][:].ravel()
        H.x = f['f2/Hx'][:].ravel()
        H.y = f['f2/Hy'][:].ravel()
        H.z = f['f2/Hz'][:].ravel()
    return face, face_n, H
    
class simple_Lens():
    def __init__(self,
                 n,thickness, D,
                 surface_file1, surface_file2,
                 widget,
                 coord_sys,
                 name = 'simplelens',
                 Device = T.device('cuda'),
                 outputfolder = 'output/'):
        self.name = name # lens name
        self.outfolder = outputfolder
        self.n = n # refractive index of the lens
        self.t = thickness # tickness of the lens in center.
        self.diameter = D # diameters
        self.surf_fnc1 = read_rsf2(surface_file1,units= 'cm')
        self.surf_fnc2 = read_rsf2(surface_file2,units= 'cm')
        self.coord_sys = coord_sys

        # define surface for sampling or for 3Dview
        self.f1 = Coord()
        self.f1_n = Coord()
        self.f2 = Coord()
        self.f2_n = Coord()
        # 3D view
        # Analysis method
        self.method = None
        self.target_face = None
        self.widget = widget
        self.surf_cur_file = None
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
        
    def PO_analysis(self,N1,N2,feed,k,
                     sampling_type_f1='rectangle',
                     phi_type_f1 = 'uniform',
                     sampling_type_f2='rectangle',
                     phi_type_f2 = 'uniform',
                     Method ='popo',
                     device = T.device('cuda'),
                     po_name = '_po_cur.h5'):
        if Method.lower() == 'popo':
            method = lensPO 
        '''sampling the model'''
        f1,f1_n = self.sampling(N1,self.surf_fnc1,self.r1,
                                          Sampling_type = sampling_type_f1,
                                          phi_type=phi_type_f1)
        
        #f1_n =scalarproduct(1,f1_n)
        
        f2,f2_n = self.sampling(N2,self.surf_fnc2,self.r2,
                                         Sampling_type = sampling_type_f2,
                                         phi_type=phi_type_f2)
        self.f2 = copy.copy(f2)
        self.f2_n = copy.copy(f2_n)
        f2 = local2global([np.pi,0,0], [0,0,self.t],f2)
        f2_n = local2global([np.pi,0,0],[0,0,0],f2_n)
        # convert two surfaces into target coordinates
        f1_p = copy.copy(f1)
        f1_p_n = copy.copy(f1_n)
        f1_p.x,f1_p.y,f1_p.z = self.coord_sys._toGlobal_coord(f1_p.x,f1_p.y,f1_p.z)
        f1_p.x,f1_p.y,f1_p.z = feed.coord_sys.Global_to_local(f1_p.x,f1_p.y,f1_p.z)
        f1_p_n.x,f1_p_n.y,f1_p_n.z = self.coord_sys._toGlobal_coord(f1_p_n.x,f1_p_n.y,f1_p_n.z)
        f1_p_n.x,f1_p_n.y,f1_p_n.z = feed.coord_sys.Global_to_local(f1_p_n.x,f1_p_n.y,f1_p_n.z)
        data = np.matmul(np.matmul(feed.coord_sys.mat_g_l,self.coord_sys.mat_l_g),
                         np.array([f1_p_n.x,f1_p_n.y,f1_p_n.z]))
        f1_p_n.x = data[0,:]
        f1_p_n.y = data[1,:]
        f1_p_n.z = data[2,:]
        
        
        '''get field on surface 1 !!!!'''
        E_in, H_in,= feed.source(f1_p,f1_p_n)
        print(np.matmul(self.coord_sys.mat_g_l,feed.coord_sys.mat_l_g))
        E_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,feed.coord_sys.mat_l_g))
        H_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,feed.coord_sys.mat_l_g))
        
        self.f_E_in = E_in
        self.f_H_in = H_in
        del(f1_p,f1_p_n)
        print('input power')
        print('poynting value max!')
        p_n = poyntingVector(self.f_E_in,self.f_H_in)
        print(abs_v(p_n).max())
        #print((abs_v(p_n)*f1.w).sum())
        print(k**2*(abs_v(p_n)*f1.w).sum())
        print('******')
        #print((1**2*epsilon*f1.w*np.abs(self.f_E_in.x)**2).sum())
        #print((mu*f1.w*np.abs(self.f_H_in.y)**2).sum())
        
        '''double PO analysis!!!'''
        self.f2_E,self.f2_H, self.f2_E_t, self.f2_E_r, self.f2_H_t,\
        self.f2_H_r, self.f1_E_t, self.f1_E_r,  self.f1_H_t , self.f1_H_r, T1, R1, T2, R2 = method(f1,f1_n,f1.w,
                                                                                        f2,f2_n,
                                                                                        self.f_E_in,self.f_H_in,
                                                                                        k,self.n,
                                                                                        device = device)
        print('Transform f1')
        print('poynting value max!')
        p_n = poyntingVector(self.f1_E_t,self.f1_H_t)
        print(abs_v(p_n).max())
        #print((abs_v(p_n)*f1.w).sum())
        print(k**2*(abs_v(p_n)*f1.w).sum())
        print('f2')
        print('poynting value max!')
        p_n = poyntingVector(self.f2_E,self.f2_H)
        print(abs_v(p_n).max())
        #print((abs_v(p_n)*f2.w).sum())
        print(k**2*(abs_v(p_n)*f2.w).sum())
        print('f2 transmission')
        print('poynting value max!')
        p_n = poyntingVector(self.f2_E_t,self.f2_H_t)
        #print((abs_v(p_n)*f2.w).sum())
        print(k**2*(abs_v(p_n)*f2.w).sum())

        # save to h5 data file
        self.surf_cur_file = self.outfolder + self.name + po_name
        with h5py.File(self.surf_cur_file,'w') as file:
            #self.f2_E_t.x = self.f2_E_t.x.reshape(N2[2],N2[0]) 
            #self.f2_E_t.y = self.f2_E_t.y.reshape(N2[2],N2[0]) 
            #self.f2_E_t.z = self.f2_E_t.z.reshape(N2[2],N2[0]) 
            #self.f2_H_t.x = self.f2_H_t.x.reshape(N2[2],N2[0]) 
            #self.f2_H_t.y = self.f2_H_t.y.reshape(N2[2],N2[0]) 
            #self.f2_H_t.z = self.f2_H_t.z.reshape(N2[2],N2[0]) 
            saveh5_surf(file,f1,f1_n, self.f_E_in, self.f_H_in,T1,R1,name = 'f1')
            saveh5_surf(file,f2,f2_n, self.f2_E_t, self.f2_H_t,T2,R2,name = 'f2')

    def source(self,
               target,
               k,
               far_near = 'near'):
        # read the source on surface face2;
        face2, face2_n, H2 = read_cur(self.surf_cur_file)
        if isinstance(target,Spherical_grd):
            face2.x,face2.y,face2.z = self.coord_sys._toGlobal_coord(face2.x,face2.y,face2.z)
            face2.x,face2.y,face2.z = target.coord_sys.Global_to_local(face2.x,face2.y,face2.z)

            data = np.matmul(np.matmul(target.coord_sys.mat_g_l,self.coord_sys.mat_l_g),
                         np.array([face2_n.x,face2_n.y,face2_n.z]))
            face2_n.x = data[0,:]
            face2_n.y = data[1,:]
            face2_n.z = data[2,:]
            H2.tocoordsys(matrix = np.matmul(target.coord_sys.mat_g_l,self.coord_sys.mat_l_g))
            print(np.matmul(target.coord_sys.mat_g_l,self.coord_sys.mat_l_g))

            
            if far_near.lower() == 'far':
                target.E,target.H = PO_far_GPU(face2,face2_n,face2.w,
                                               target.grid,
                                               0,
                                               H2,
                                               k,
                                               device =T.device('cuda'))
            else:
                target.E,target.H = PO_GPU(face2,face2_n,face2.w,
                                           target.grid,
                                           0,
                                           H2,
                                           k,
                                           1, # n refractive index
                                           device =T.device('cuda'))
        else:
            pass
    ## sampling technique
    def sampling(self,
                 f1_N, surf_fuc,r1,r0=0,
                 Sampling_type = 'polar',
                 phi_type = 'uniform'):
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
