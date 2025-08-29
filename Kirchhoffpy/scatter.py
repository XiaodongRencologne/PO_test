import os
import numpy as np
import copy
import torch as T

from .coordinate_operations import Coord;

from .field_storage import Spherical_grd
from .POpyGPU import PO_far_GPU2 as PO_far_GPU
from .POpyGPU import epsilon,mu
from .POpyGPU import PO_GPU_2 as PO_GPU

from .Vopy import vector,abs_v,scalarproduct, CO
from .RWcur import saveh5_surf,read_cur

import pyvista as pv
pv.set_jupyter_backend('trame')


class scatter():
    def __init__(self,
                 coord_sys,
                 rim,
                 surface,
                 name = 'scatter',
                 outputfolder = 'output/'):
        self.coord_sys = coord_sys
        self.rim = rim
        self.surface = surface
        self.name = name
        self.outfolder = outputfolder
        
        self.face =  Coord()
        self.face.x, self.face.y, self.face.w = self.rim.sampling(po1,po2)
        self.face.z, _ = self.surface.surface(self.face.x,self.face.y)



    def get_current(self,
                    source,k,
                    po1 =10 ,po2 = 10,
                    guadrature = 'gaussian',
                    Phi_type = 'less',
                    po_name = '_po_cur.h5',
                    device = T.device('cuda')):

        face = Coord()

        if isinstance(self.rim, Elliptical_rim):
            face.x, face.y, face.w= self.rim.sampling(po1,po2,
                                                    quadrature = quadrature,
                                                    Phi_type = Phi_type)
            
        else:
            face.x, face.y, face.w = self.rim.sampling(po1,po2, quadrature = quadrature)
        # get the sampling on the scatter surface
        face.z, face_n = self.surface.surface(face.x,face.y)

        # copy the surface, and convert the face to the coordinate system of the source coord
        face_p = copy.copy(face)
        face_p.x,face_p.y,face_p.z = self.coord_sys.To_coord_sys(source.coord_sys,
                                                                 face.x, face.y, face.z)

        # get the field on the surface
        self.E_in, self.H_in, = source.source(face_p,k,device = device)
        # conver the field to the coordinate system of the scatter
        self.E_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,source.coord_sys.mat_l_g))
        self.H_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,source.coord_sys.mat_l_g))

        self.surf_cur_file = self.outfolder + self.name + po_name
        
        # save to the current file
        saveh5_surf(self.surf_cur_file,face,face_n, self.E_in, self.H_in,0,0,name = 'f2')

    def source(self,
               target,k,
               n=1,
               far_near = 'near',
               device = T.device('cuda')):        
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
                                           n, # n refractive index
                                           device = device)
            return target.E, target.H
        else:
            print('Here')
            E, H = PO_GPU(face2,face2_n,face2.w,
                                        target.face,
                                        E2,
                                        H2,
                                        k,
                                        n, # n refractive index
                                        device =device)
            return E,H



class scatter_lens():
    def __init__(self,
                 coord_sys,
                 rim,
                 surface,
                 ARfile = None,
                 name = 'scatter',
                 outputfolder = 'output/'):
        self.coord_sys = coord_sys
        self.rim = rim
        self.surface = surface
        self.name = name
        self.outfolder = outputfolder
        
        self.face =  Coord()
        self.face.x, self.face.y, self.face.w = self.rim.sampling(po1,po2)
        self.face.z, _ = self.surface.surface(self.face.x,self.face.y)



    def get_current(self,
                    source,k,
                    po1 =10 ,po2 = 10,
                    guadrature = 'gaussian',
                    Phi_type = 'less',
                    po_name = '_po_cur.h5',
                    device = T.device('cuda')):

        face = Coord()

        if isinstance(self.rim, Elliptical_rim):
            face.x, face.y, face.w= self.rim.sampling(po1,po2,
                                                    quadrature = quadrature,
                                                    Phi_type = Phi_type)
            
        else:
            face.x, face.y, face.w = self.rim.sampling(po1,po2, quadrature = quadrature)
        # get the sampling on the scatter surface
        face.z, face_n = self.surface.surface(face.x,face.y)

        # copy the surface, and convert the face to the coordinate system of the source coord
        face_p = copy.copy(face)
        face_p.x,face_p.y,face_p.z = self.coord_sys.To_coord_sys(source.coord_sys,
                                                                 face.x, face.y, face.z)

        # get the field on the surface
        self.E_in, self.H_in, = source.source(face_p,k,device = device)
        # conver the field to the coordinate system of the scatter
        self.E_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,source.coord_sys.mat_l_g))
        self.H_in.tocoordsys(matrix = np.matmul(self.coord_sys.mat_g_l,source.coord_sys.mat_l_g))

        self.surf_cur_file = self.outfolder + self.name + po_name
        
        # save to the current file
        saveh5_surf(self.surf_cur_file,face,face_n, self.E_in, self.H_in,0,0,name = 'f2')

    def source(self,
               target,k,
               n=1,
               far_near = 'near',
               device = T.device('cuda')):        
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
                                           n, # n refractive index
                                           device = device)
            return target.E, target.H
        else:
            print('Here')
            E, H = PO_GPU(face2,face2_n,face2.w,
                                        target.face,
                                        E2,
                                        H2,
                                        k,
                                        n, # n refractive index
                                        device =device)
            return E,H




def autoconvergence(scatter,
                    source,
                    target,
                    accuracy = -80):
    Accuracy = 10**(accuracy/20)
    def method(N):
        scatter.get_current(source,k,
                            po1 =N ,po2 = 10,
                            guadrature = 'gaussian',
                            Phi_type = 'less',
                            po_name = '_po_cur.h5',
                            device = T.device('cuda'))
        E,_ = scatter.source(target, k ,
                                n=1,
                                far_near = 'near',
                                device = T.device('cuda'))
        return abs_v(E)
        
    po1, loops1, status1 = try_convergence(method,
                                            10, 
                                            Accuracy)
    del(method)
    def method(N):
        scatter.get_current(source,k,
                            po1 =10 ,po2 = N,
                            guadrature = 'gaussian',
                            Phi_type = 'less',
                            po_name = '_po_cur.h5',
                            device = T.device('cuda'))
        E,_ = scatter.source(target, k ,
                                n=1,
                                far_near = 'near',
                                device = T.device('cuda'))
        return abs_v(E)
        
    po2, loops2, status2 = try_convergence(method,
                                            10, 
                                            Accuracy)
    
    return po1,po2