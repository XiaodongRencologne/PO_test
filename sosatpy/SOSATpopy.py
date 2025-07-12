import pyvista as pv
from .Kirchhoffpy import lenspy
from .Kirchhoffpy import Feedpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from .Kirchhoffpy import coordinate,field_storage
from .Kirchhoffpy.Vopy import CO,dotproduct,vector
import torch as T
import h5py

from polarizationpy import polar_angle
c=299792458
p = pv.Plotter()
srffolder = 'srf/'

class sosat:
    def __init__(self, 
                 freq,
                 taper_A = 10.1161095,
                 edge_taper = -2.1714724,
                 feedfile = None,
                 feedpos = [0,0,0], # dx, dy, dz
                 feedrot = [0,0,0],  # rotation along x-aixs, y-axis and z-axis
                 polarization = 'x',
                 AR_file = None, # data of AR coating, Fresnel coefficien given by sqaure root of the coefficents in power.
                 groupname = None, # normally the name is the frequency
                 outputfolder = ''
                 ):
        self.freq = freq
        self.Lambda = c*1000 / (freq*10**9)
        self.k = 2 * np.pi / self.Lambda
        self.outputfolder = outputfolder
        self.feedpos = feedpos
        self.feedrot = feedrot
        ## 1.  define coordinate system
        dx, dy, dz = feedpos[0], feedpos[1], feedpos[2]
        dAx, dAy, dAz = feedrot[0], feedrot[1], feedrot[2]
        eff_focal_length = 569.56 #mm
        coord_ref = coordinate.coord_sys([0,0,0],[0,0,0],axes = 'xyz')
        coord_L1 = coordinate.coord_sys([0,0,-(803.9719951339465-4.34990822154231*10)],[np.pi,0,0],axes = 'xyz',ref_coord = coord_ref)
        coord_L2 = coordinate.coord_sys([0,0,-(227.64396727901004-4.696706712699847*10)],[np.pi,0,0],axes = 'xyz',ref_coord = coord_ref)
        coord_L3 = coordinate.coord_sys([0,0,-(71.77590111674095-2.96556*10)],[np.pi,0,0],axes = 'xyz',ref_coord = coord_ref)

        coord_feed_offset = coordinate.coord_sys([dx,dy,dz],[np.pi,0,0],axes = 'xyz',ref_coord = coord_ref)
        coord_feed_rotation = coordinate.coord_sys([0,0,0],[0,0,dAz*np.pi/180],axes = 'xyz',ref_coord = coord_feed_offset)
        coord_feed = coordinate.coord_sys([0,0,0],[0,0,0],axes = 'xyz',ref_coord = coord_feed_rotation)
        coord_sky_ref = coordinate.coord_sys([0,0,0],[np.pi,0,0],axes = 'xyz',ref_coord = coord_ref)
        coord_sky = coordinate.coord_sys([0,0,0],[0,0,0],axes = 'xyz',ref_coord = coord_sky_ref)

        # 2. define input Feedhorn
        self.feed= Feedpy.GaussiBeam(edge_taper ,
                                taper_A,
                                self.k,
                                coord_feed,
                                polarization = 'x')
        
        # 3. define lens
        self.L1 = lenspy.simple_Lens(3.36,
                                4.34991*10,# Thickness
                                44.8*10, # diameter
                                srffolder + 'lens1_f2.rsf', 
                                srffolder + 'lens1_f1.rsf',
                                p,
                                coord_L1,
                                name = 'L1',
                                AR_file = AR_file,
                                groupname = groupname,
                                outputfolder = outputfolder)
        self.L2 = lenspy.simple_Lens(3.36,
                                4.69671*10,# Thickness
                                44.8*10, # diameter
                                srffolder + 'lens2_f2.rsf', 
                                srffolder + 'lens2_f1.rsf',
                                p,
                                coord_L2,
                                name = 'L2',
                                AR_file = AR_file,
                                groupname = groupname,
                                outputfolder = outputfolder)

        self.L3 = lenspy.simple_Lens(3.36,
                                2.96556*10,# Thickness
                                44.8*10, # diameter
                                srffolder + 'lens3_f2.rsf', 
                                srffolder + 'lens3_f1.rsf',
                                p,
                                coord_L3,
                                name = 'L3',
                                AR_file = AR_file,
                                groupname = groupname,
                                outputfolder = outputfolder)
        
        # 4 define field grids in far-field region or near-field region
        self.center_grd = field_storage.Spherical_grd(coord_sky,
                                                -np.arctan(dx/569.56),
                                                -np.arctan(dy/569.56),
                                                10/180*np.pi,
                                                10/180*np.pi,
                                                501,501,
                                                Type = 'uv', 
                                                far_near = 'far',
                                                distance = 50000)
        self.center_grd.grid.x = self.center_grd.grid.x.ravel()
        self.center_grd.grid.y = self.center_grd.grid.y.ravel()
        self.center_grd.grid.z = self.center_grd.grid.z.ravel()
        self.field_grid_fname = self.outputfolder + str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+ '_grd.h5'
    def run_po(self,
               L3_N,
               L2_N,
               L1_N):
        L3_N1 = L3_N[0]
        L3_N2 = L3_N[1]
        L2_N1 = L2_N[0]
        L2_N2 = L2_N[1]
        L1_N1 = L1_N[0]
        L1_N2 = L1_N[1]
        # start po analysis
        dx, dy, dz = self.feedpos[0], self.feedpos[1], self.feedpos[2]
        dAx, dAy, dAz = self.feedrot[0], self.feedrot[1], self.feedrot[2]
        self.L3.PO_analysis([1,L3_N1[0],L3_N1[1],1],
                            [1,L3_N2[0],L3_N2[1],1],
                            self.feed,self.k,
                            sampling_type_f1='polar',
                            phi_type_f1 = 'less',
                            sampling_type_f2='polar',
                            phi_type_f2 = 'less',
                            po_name = '_po_cur_'+str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+'deg.h5',
                            Method ='POPO')
        self.L2.PO_analysis([1,L2_N1[0],L2_N1[1],1],
                            [1,L2_N2[0],L2_N2[1],1],
                            self.L3,self.k,
                            sampling_type_f1='polar',
                            phi_type_f1 = 'less',
                            sampling_type_f2='polar',
                            phi_type_f2 = 'less',
                            po_name = '_po_cur_'+str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+'deg.h5',
                            Method ='POPO')
        self.L1.PO_analysis([1,L1_N1[0],L1_N1[1],1],
                            [1,L1_N2[0],L1_N2[1],1],
                            self.L2,self.k,
                            sampling_type_f1='polar',
                            phi_type_f1 = 'less',
                            sampling_type_f2='polar',
                            phi_type_f2 = 'less',
                            po_name = '_po_cur_'+str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+'deg.h5',
                            Method ='POPO')
        self.L1.source(self.center_grd,
                       self.k,
                       far_near = 'far')
        
        field_storage.save_grd(self.center_grd, self.field_grid_fname)
        #self.plot_beam()
    def plot_beam(self,field_name = None, output_picture_name = 'co_cx_rot_beam.png'):
        if field_name == None:
            field_name = self.field_grid_fname
        dx, dy, dz = self.feedpos[0], self.feedpos[1], self.feedpos[2]
        dAx, dAy, dAz = self.feedrot[0], self.feedrot[1], self.feedrot[2]
        picture_fname1 = self.outputfolder +str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+output_picture_name
        picture_fname1 = self.outputfolder +str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+'rotated_'+output_picture_name
        x, y, Ex, Ey, Ez = field_storage.read_grd(field_name)
        E = vector()
        E.x = Ex
        E.y = Ey
        E.z = Ez

        r, theta, phi = self.center_grd.coord_sys._toSpherical(self.center_grd.grid.x,
                                                               self.center_grd.grid.y,
                                                               self.center_grd.grid.z)
        co,cx,crho = CO(theta,phi)
        E_co = dotproduct(E,co)
        E_cx = dotproduct(E,cx)
        power_beam = np.abs(E_co)**2 + np.abs(E_cx)**2 
        peak = power_beam.max()
        NN = np.where((np.abs(E_co)**2 + np.abs(E_cx)**2 )/peak > 10**(-15/10))[0]
        r= polar_angle.polarization_angle(np.concatenate((E_co[NN],E_cx[NN])).reshape(2,-1))
        print('rotation angle method 3: ',r.x*180/np.pi-dAz, r.status)
        Beam_new = polar_angle.rotation_angle(r.x,np.concatenate((E_co,E_cx)).reshape(2,-1))
        E_co_new = Beam_new[0,:]
        E_cx_new = Beam_new[1,:]

        Nx = x.size
        Ny = y.size
        vmax1 = np.abs(E_co).max()
        vmax2 = np.abs(E_cx).max()
        vmax = np.log10(max(vmax1, vmax2))*20
        fig, ax = plt.subplots(1,2, figsize=(12, 6))
        fig.suptitle('Beams at '+str(dx)+'mm,'+'Polar: ' + str(dAz))
        p1 = ax[0].pcolor(x,y, 10*np.log10(np.abs(E_co.reshape(Ny,Nx))**2),vmax = vmax,vmin= vmax-60)
        ax[0].set_title('co-polar beam')
        p2 = ax[1].pcolor(x,y, 10*np.log10(np.abs(E_cx.reshape(Ny,Nx))**2),vmax = vmax,vmin= vmax-60)
        ax[1].set_title('cx-polar beam')
        cbar = fig.colorbar(p1, ax=[ax[0], ax[1]], orientation='vertical', fraction=0.05, pad=0.1)
        plt.savefig(picture_fname1, dpi=300)
        plt.show()

        vmax1 = np.abs(E_co_new).max()
        vmax2 = np.abs(E_cx_new).max()
        vmax = np.log10(max(vmax1, vmax2))*20
        fig, ax = plt.subplots(1,2, figsize=(12, 6))
        fig.suptitle('Beams rotated by'+str(r.x[0]*180/np.pi)+' deg')
        p3 = ax[0].pcolor(x,y, 10*np.log10(np.abs(E_co_new.reshape(Ny,Nx))**2),vmax = vmax,vmin= vmax-60)
        p4 = ax[1].pcolor(x,y, 10*np.log10(np.abs(E_cx_new.reshape(Ny,Nx))**2),vmax = vmax,vmin= vmax-60)
        cbar = fig.colorbar(p3, ax=[ax[0], ax[1]], orientation='vertical', fraction=0.05, pad=0.1)
        plt.savefig(picture_fname1, dpi=300)
        plt.show()
