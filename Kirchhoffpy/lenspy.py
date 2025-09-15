
#%%
"""
Package used to build mirror model based on 'uniform sampling' and 'Gaussian quadrature';

"""
import os
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

from .LensPO import Z0,lensPO,lensPO_AR
from .POpyGPU import PO_far_GPU2 as PO_far_GPU
from .POpyGPU import epsilon,mu
from .POpyGPU import PO_GPU_2 as PO_GPU

from .Vopy import vector,abs_v,scalarproduct, CO
from .RWcur import saveh5_surf,read_cur
from .FresnelCoeff import poyntingVector

import pyvista as pv
pv.set_jupyter_backend('trame')#('static')#

import matplotlib.pyplot as plt
#%%
def read_rsf(file,units= 'cm'):
    factor= 1.0
    if units == 'cm':
        factor = 10
    elif units == 'mm':
        factor = 1.0
    elif units == 'm':
        factor = 1000
    data = np.genfromtxt(file, skip_header = 2)
    r = data[:,0]*factor
    z = data[:,1]*factor
    cs = CubicSpline(r,z)
    cs_derivative = cs.derivative()
    
    def srf(x,y):
        r = np.sqrt(x**2+y**2)
        z = cs(r)
        n = Coord()
        # surface normal vector
        r = np.where(r == 0, 10**(-10), r)
        n.z = -np.ones(x.shape)
        n.x = cs_derivative(r)*x/r
        n.y = cs_derivative(r)*y/r
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
    r = (data[:,0]*factor)**2
    z = data[:,1]*factor
    cs = CubicSpline(r,z)
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
                 coord_sys,
                 units = 'cm',
                 name = 'simplelens',
                 AR_file = None,
                 groupname = None,
                 Device = T.device('cuda'),
                 outputfolder = 'output/'):
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        self.AR_file = AR_file
        self.groupname = groupname
        self.name = name # lens name
        self.outfolder = outputfolder
        self.n = n # refractive index of the lens
        self.t = thickness # tickness of the lens in center.
        self.diameter = D # diameters
        self.surf_fnc1 = read_rsf(surface_file1,units= units)
        self.surf_fnc2 = read_rsf(surface_file2,units= units)
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
    def r1(self,theta):
        x0=self.x1
        y0=self.y1
        R = self.diameter/2
        return -(x0*np.cos(theta)+y0*np.sin(theta)) + np.sqrt(R**2-(x0*np.sin(theta)-y0*np.cos(theta))**2)
        #return np.ones(theta.shape)*self.diameter/2
    def r2(self,theta):
        x0=self.x2
        y0=self.y2
        R = self.diameter/2
        return -(x0*np.cos(theta)+y0*np.sin(theta)) + np.sqrt(R**2-(x0*np.sin(theta)-y0*np.cos(theta))**2)

    def convergence_test(self,
                         feed,
                         k):
        pass   
    def PO_analysis(self,N1,N2,source,k,
                     sampling_type_f1='rectangle',
                     phi_type_f1 = 'uniform',
                     sampling_type_f2='rectangle',
                     phi_type_f2 = 'uniform',
                     Method ='popo',
                     device = T.device('cuda'),
                     po_name = '_po_cur.h5',
                     x0=0,y0=0,
                     convergence_test = True):
        self.x1 =x0
        self.y1 =y0
        self.x2 =x0
        self.y2 =y0
        
        if Method.lower() == 'popo':
            if self.AR_file is not None:
                method = lensPO_AR
            else:
                method = lensPO
        elif Method.lower() == 'gopo':
            pass
        elif Method.lower() == 'A-po':
            pass
        elif Method.lower() == 'double-po':
            pass
        '''sampling the model'''
        f1,f1_n = self.sampling(N1,self.surf_fnc1,self.r1,
                                          Sampling_type = sampling_type_f1,
                                          phi_type=phi_type_f1,x0 = x0,y0 = y0)
        
        f1_n =scalarproduct(-1,f1_n)
        
        f2,f2_n = self.sampling(N2,self.surf_fnc2,self.r2,
                                         Sampling_type = sampling_type_f2,
                                         phi_type=phi_type_f2,x0 = x0,y0 = y0)
        #f2_n =scalarproduct(-1,f2_n)
        self.f2 = copy.copy(f2)
        self.f2_n = copy.copy(f2_n)
        f2 = local2global([np.pi,0,0], [0,0,self.t],f2)
        f2_n = local2global([np.pi,0,0],[0,0,0],f2_n)
        # convert two surfaces into target coordinates
        f1_p = copy.copy(f1)
        f1_p_n = copy.copy(f1_n)
        f1_p.x,f1_p.y,f1_p.z = self.coord_sys.Local_to_Global(f1_p.x,f1_p.y,f1_p.z)
        f1_p.x,f1_p.y,f1_p.z = source.coord_sys.Global_to_Local(f1_p.x,f1_p.y,f1_p.z)
        #f1_p_n.x,f1_p_n.y,f1_p_n.z = self.coord_sys._toGlobal_coord(f1_p_n.x,f1_p_n.y,f1_p_n.z)
        #f1_p_n.x,f1_p_n.y,f1_p_n.z = feed.coord_sys.Global_to_local(f1_p_n.x,f1_p_n.y,f1_p_n.z)
        data = np.matmul(np.matmul(source.coord_sys.mat_g_l,self.coord_sys.mat_l_g),
                         np.array([f1_p_n.x,f1_p_n.y,f1_p_n.z]))
        f1_p_n.x = data[0,:]
        f1_p_n.y = data[1,:]
        f1_p_n.z = data[2,:]
        
        
        '''get field on surface 1 !!!!'''
        E_in, H_in,= source.source(f1_p,k)
        #print(np.matmul(self.coord_sys.mat_g_l,feed.coord_sys.mat_l_g))

        '''convert the field to the scatter local coordinate system'''
        E_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,source.coord_sys.mat_l_g))
        H_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,source.coord_sys.mat_l_g))
        self.f_E_in = E_in
        self.f_H_in = H_in
        del(f1_p,f1_p_n)
        """
        print('input power')
        print('poynting value max!')
        p_n = poyntingVector(self.f_E_in,self.f_H_in)
        print(abs_v(p_n).max())
        #print((abs_v(p_n)*f1.w).sum())
        print(k**2*(abs_v(p_n)*f1.w).sum())
        print('******')
        #print((1**2*epsilon*f1.w*np.abs(self.f_E_in.x)**2).sum())
        #print((mu*f1.w*np.abs(self.f_H_in.y)**2).sum())

        """

        '''double PO analysis!!!'''
        if self.AR_file is not None:
            self.f2_E,self.f2_H, self.f2_E_t, self.f2_E_r, self.f2_H_t,\
            self.f2_H_r, self.f1_E_t, self.f1_E_r,  self.f1_H_t , self.f1_H_r, T1, R1, T2, R2 = method(f1,f1_n,f1.w,
                                                                                            f2,f2_n,
                                                                                            self.f_E_in,self.f_H_in,
                                                                                            k,self.n,
                                                                                            self.AR_file,
                                                                                            self.groupname,
                                                                                            device = device)
        else:
            self.f2_E,self.f2_H, self.f2_E_t, self.f2_E_r, self.f2_H_t,\
            self.f2_H_r, self.f1_E_t, self.f1_E_r,  self.f1_H_t , self.f1_H_r, T1, R1, T2, R2 = method(f1,f1_n,f1.w,
                                                                                            f2,f2_n,
                                                                                            self.f_E_in,self.f_H_in,
                                                                                            k,self.n,
                                                                                            device = device)
    
        """
        print('Transform f1')
        print('poynting value max!')
        p_n = poyntingVector(self.f1_E_t,self.f1_H_t)    
        """
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
        """
        print(abs_v(p_n).max())
        #print((abs_v(p_n)*f2.w).sum())
        print(k**2*(abs_v(p_n)*f2.w).sum())
        print('f2 transmission')
        print('poynting value max!')
        p_n = poyntingVector(self.f2_E_t,self.f2_H_t)
        #print((abs_v(p_n)*f2.w).sum())
        print(k**2*(abs_v(p_n)*f2.w).sum())
        """

        # save to h5 data file
        self.surf_cur_file = self.outfolder + self.name + po_name
        with h5py.File(self.surf_cur_file,'w') as file:
            #self.f2_E_t.x = self.f2_E_t.x.reshape(N2[2],N2[0]) 
            #self.f2_E_t.y = self.f2_E_t.y.reshape(N2[2],N2[0]) 
            #self.f2_E_t.z = self.f2_E_t.z.reshape(N2[2],N2[0]) (
            saveh5_surf(file,f1,f1_n, self.f_E_in, self.f_H_in,T1,R1,name = 'f1')
            saveh5_surf(file,f2,f2_n, self.f2_E_t, self.f2_H_t,T2,R2,name = 'f2')

    def source(self,
               target,k,
               far_near = 'near',
               device = T.device('cuda'),
               cur_file = None):
        # read the source on surface face2;
        if cur_file == None:
            face2, face2_n, H2, E2= read_cur(self.surf_cur_file)
        else:
            face2, face2_n, H2, E2= read_cur(cur_file)        
        if isinstance(target,Spherical_grd):
            face2.x,face2.y,face2.z = self.coord_sys.Local_to_Global(face2.x,face2.y,face2.z)
            face2.x,face2.y,face2.z = target.coord_sys.Global_to_Local(face2.x,face2.y,face2.z)

            data = np.matmul(np.matmul(target.coord_sys.mat_g_l,self.coord_sys.mat_l_g),
                         np.array([face2_n.x,face2_n.y,face2_n.z]))
            face2_n.x = data[0,:]
            face2_n.y = data[1,:]
            face2_n.z = data[2,:]
            H2.tocoordsys(matrix = np.matmul(target.coord_sys.mat_g_l,self.coord_sys.mat_l_g))
            E2.tocoordsys(matrix = np.matmul(target.coord_sys.mat_g_l,self.coord_sys.mat_l_g))
            print(np.matmul(target.coord_sys.mat_g_l,self.coord_sys.mat_l_g))

            #grid = copy.copy(target.grid)
            #grid.x, grid.y, grid.z = target.coord_sys._toGlobal_coord(target.grid.x,target.grid.y,target.grid.z)
            if far_near.lower() == 'far':
                print('*(**)')
                target.E,target.H = PO_far_GPU(face2,face2_n,face2.w,
                                               target.grid,
                                               E2,
                                               H2,
                                               k,
                                               device = device)
            else:
                target.E,target.H = PO_GPU(face2,face2_n,face2.w,
                                           target.grid,
                                           E2,
                                           H2,
                                           k,
                                           1, # n refractive index
                                           device =T.device('cuda'))
        else:
            print('Here')
            E,H = PO_GPU(face2,face2_n,face2.w,
                                        target,
                                        E2,
                                        H2,
                                        k,
                                        1, # n refractive index
                                        device =T.device('cuda'))
            return E, H
    ## sampling technique
    def sampling(self,
                 f1_N, surf_fuc,r1,r0=0,
                 Sampling_type = 'polar',
                 phi_type = 'uniform',x0 = 0, y0 = 0):
        '''
        sampling_type = 'Gaussian' / 'uniform'
        '''
        f1 = Coord()
        #f2 = Coord()

        if Sampling_type == 'polar':
            print('sampling type :', phi_type)
            f1.x, f1.y, f1.w= Guass_L_quadrs_Circ(0,r1,
                                        f1_N[0],f1_N[1],
                                        0,2*np.pi,f1_N[2],
                                        Phi_type=phi_type)
            f1.x = f1.x + x0
            f1.y = f1.y + y0
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
# %%
