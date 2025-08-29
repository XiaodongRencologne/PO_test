import os
import numpy as np
import copy
import h5py
import torch as T

#from .Gauss_L_quadr import Guass_L_quadrs_Circ,Gauss_L_quadrs2d
from .coordinate_operations import Coord;
from .rim import Elliptical_rim
#from .coordinate_operations import Transform_local2global as local2global;
#from .coordinate_operations import Transform_global2local as global2local;
from .field_storage import Spherical_grd

from .POpyGPU import PO_far_GPU2 as PO_far_GPU
from .POpyGPU import epsilon,mu
from .POpyGPU import PO_GPU_2 as PO_GPU

from .Vopy import vector,abs_v,scalarproduct, CO
from .RWcur import saveh5_surf,read_cur

import pyvista as pv
pv.set_jupyter_backend('trame')#('static')#

import matplotlib.pyplot as plt



class Aperture():
    def __init__(self,
                 coord_sys,
                 rim,
                 name = 'Aperture',
                 outputfolder = 'output/'):
        self.coord_sys = coord_sys
        self.rim = rim
        self.name = name
        self.outfolder = outputfolder
    def get_current(self,
                    source,k,
                    po1 =0 ,po2 = 0,
                    quadrature = 'gaussian',
                    Phi_type = 'less',
                    po_name = '_po_cur.h5'):
        points  = Coord()
        points_n = Coord()
        if isinstance(self.rim, Elliptical_rim):
            points.x, points.y, points.w= self.rim.sampling(po1,po2,
                                                    quadrature = quadrature,
                                                    Phi_type = Phi_type)
            
        else:
            points.x, points.y, points.w = self.rim.sampling(po1,po2, quadrature = quadrature)
        
        points.z =np.zeros(points.x.shape)
        points_n.x = np.zeros(points.x.shape)
        points_n.y = np.zeros(points.x.shape)
        points_n.z = np.ones(points.x.shape)
        points_n.N = np.ones(points.x.shape)

        points_p = copy.copy(points)
        points_p.x,points_p.y,points_p.z = self.coord_sys.To_coord_sys(source.coord_sys, 
                                                                        points.x, points.y, points.z)

        E_in, H_in,= source.source(points_p,k)
        
        E_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,source.coord_sys.mat_l_g))
        H_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,source.coord_sys.mat_l_g))

        self.surf_cur_file = self.outfolder + self.name + po_name
        with h5py.File(self.surf_cur_file,'w') as file:
            saveh5_surf(file,points,points_n, E_in, H_in,0,0,name = 'f2')

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


        
class Filter(Aperture):
    def __init__(self,
                 coord_sys,
                 rim,
                 AR_file = None,
                 groupname = None,
                 name = 'filter',
                 outputfolder = 'output/'):
            super().__init__(coord_sys,
                                rim,
                                name,
                                outputfolder)
            self.AR_file = AR_file
            if self.AR_file == None:
                print('Perfect Transmission!!!')
            else:
                self.Fresnel_co1,_= read_Fresnel_coeffi_AR(self.AR_file, groupname, 1, 1)

    
    def get_current(self,
                    source,k,
                    po1 =0 ,po2 = 0,
                    quadrature = 'gaussian',
                    Phi_type = 'less',
                    po_name = '_po_cur.h5'):
        points  = Coord()
        points_n = Coord()
        if isinstance(self.rim, Elliptical_rim):
            points.x, points.y, points.w= self.rim.sampling(po1,po2,
                                                    quadrature = quadrature,
                                                    Phi_type = Phi_type)
            
        else:
            points.x, points.y, points.w = self.rim.sampling(po1,po2, quadrature = quadrature)
        
        points.z =np.zeros(points.x.shape)
        points_n.x = np.zeros(points.x.shape)
        points_n.y = np.zeros(points.x.shape)
        points_n.z = np.ones(points.x.shape)
        points_n.N = np.ones(points.x.shape)

        points_p = copy.copy(points)
        points_p.x,points_p.y,points_p.z = self.coord_sys.To_coord_sys(source.coord_sys, 
                                                                        points.x, points.y, points.z)

        E_in, H_in,= source.source(points_p,k)
        self.surf_cur_file = self.outfolder + self.name + po_name

        with h5py.File(self.surf_cur_file,'w') as file:
            
            if self.AR_file is not None:
                E_t,E_r,H_t,H_r, poynting,T,R,_ = calculate_Field_T_R_AR(1,1,points_n, E_in,H_in, self.Fresnel_co)
                saveh5_surf(file,f1,f1_n,E_in, H_in,0,0,name = 'fin')
                saveh5_surf(file,points,points_n,E_out,H_out,T,R,name = 'f2')  
            else:
                saveh5_surf(file,points,points_n,E_in,H_in,0,0,name = 'f2')  
            
            
        