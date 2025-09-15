import pyvista as pv
from Kirchhoffpy import lenspy
from Kirchhoffpy import Feedpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from Kirchhoffpy import coordinate,field_storage
from Kirchhoffpy.Vopy import CO,dotproduct,vector
import torch as T
import h5py

from polarizationpy import polar_angle,stoke_param

c=299792458
p = pv.Plotter()
srffolder = 'HDPE_LENS_srf/srf_plano_convex/'

SILICON = 3.36
HDPE = 1.525
AR = 1.23

L_lens1_Lyot = 5 #mm
L_lens1_ref = 624.7169956168777
L_lens2_ref = 170
L_Ly_ref = L_lens1_ref + L_lens1_Lyot

lens_diameter1 = 15.1*20 # mm
lens_diameter2 = 16*20 # mm

lens_thickness1 = 36.94389216703172 #mm
lens_thickness2 = 40 #mm
class HDEP_2lens():
    def __init__(self, 
                 freq,
                 taper_A = 17,
                 edge_taper = -3,
                 feedfile = None,
                 feedpos = [0,0,0], # dx, dy, dz
                 feedrot = [0,0,0],  # rotation along x-aixs, y-axis and z-axis
                 polarization = 'x',
                 grid_sizex = 0.2,
                 grid_sizey = 0.2,
                 Nx = 201,
                 Ny = 201,
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
        eff_focal_length = 479 #mm
        coord_ref = coordinate.coord_sys([0,0,0],[0,0,0],axes = 'xyz')
        coord_L1 = coordinate.coord_sys([0,0,-(L_lens1_ref - lens_thickness1)],[np.pi,0,0],axes = 'xyz',ref_coord = coord_ref)
        coord_L2 = coordinate.coord_sys([0,0,-(L_lens2_ref - lens_thickness2)],[np.pi,0,0],axes = 'xyz',ref_coord = coord_ref)

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
        self.L1 = lenspy.simple_Lens(HDPE,
                                36.94389216703172,# Thickness
                                lens_diameter1, # diameter
                                srffolder + 'lens1_plano.rsf', 
                                srffolder + 'lens1_f1.rsf',
                                p,
                                coord_L1,
                                units = 'mm',
                                name = 'L1',
                                AR_file = AR_file,
                                groupname = groupname,
                                outputfolder = outputfolder)
        self.L2 = lenspy.simple_Lens(HDPE,
                                40,# Thickness
                                lens_diameter2, # diameter
                                srffolder + 'lens2_plano.rsf', 
                                srffolder + 'lens2_f1.rsf',
                                p,
                                coord_L2,
                                units = 'mm',
                                name = 'L2',
                                AR_file = AR_file,
                                groupname = groupname,
                                outputfolder = outputfolder)
        
        # 4 define field grids in far-field region or near-field region
        self.center_grd = field_storage.Spherical_grd(coord_sky,
                                                -np.arctan(dx/eff_focal_length),
                                                np.arctan(dy/eff_focal_length),
                                                grid_sizex,
                                                grid_sizey,
                                                Nx,Ny,
                                                Type = 'uv', 
                                                far_near = 'far',
                                                distance = 50000)
        self.center_grd.grid.x = self.center_grd.grid.x.ravel()
        self.center_grd.grid.y = self.center_grd.grid.y.ravel()
        self.center_grd.grid.z = self.center_grd.grid.z.ravel()
        self.field_grid_fname = self.outputfolder + str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+ '_grd.h5'
    def run_po(self,
               L2_N,
               L1_N,
               sampling_type = 'less'):
        L2_N1 = L2_N[0]
        L2_N2 = L2_N[1]
        L1_N1 = L1_N[0]
        L1_N2 = L1_N[1]
        # start po analysis
        dx, dy, dz = self.feedpos[0], self.feedpos[1], self.feedpos[2]
        dAx, dAy, dAz = self.feedrot[0], self.feedrot[1], self.feedrot[2]
        
        self.L2.PO_analysis([1,L2_N1[0],L2_N1[1],1],
                            [1,L2_N2[0],L2_N2[1],1],
                            self.feed,self.k,
                            sampling_type_f1='polar',
                            phi_type_f1 = sampling_type ,
                            sampling_type_f2='polar',
                            phi_type_f2 = sampling_type ,
                            po_name = '_po_cur_'+str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+'deg.h5',
                            Method ='POPO')
        self.L1.PO_analysis([1,L1_N1[0],L1_N1[1],1],
                            [1,L1_N2[0],L1_N2[1],1],
                            self.L2,self.k,
                            sampling_type_f1='polar',
                            phi_type_f1 = sampling_type ,
                            sampling_type_f2='polar',
                            phi_type_f2 = sampling_type ,
                            po_name = '_po_cur_'+str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+'deg.h5',
                            Method ='POPO')
        
        self.L1.source(self.center_grd,
                       self.k,
                       far_near = 'far')
        
        field_storage.save_grd(self.center_grd, self.field_grid_fname)
        #self.plot_beam()
    def plot_beam(self,field_name = None, output_picture_name = 'co_cx_rot_beam.png',cmap = 'jet',plot = True):
        if field_name == None:
            field_name = self.field_grid_fname
        dx, dy, dz = self.feedpos[0], self.feedpos[1], self.feedpos[2]
        dAx, dAy, dAz = self.feedrot[0], self.feedrot[1], self.feedrot[2]
        picture_fname1 = self.outputfolder +str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+output_picture_name
        picture_fname2 = self.outputfolder +str(dx)+'_'+str(dy)+'_'+str(dz)+'mm_polar_'+str(dAz)+'rotated_'+output_picture_name
        x, y, Ex, Ey, Ez = field_storage.read_grd(field_name)
        E = vector()
        E.x = Ex
        E.y = Ey
        E.z = Ez

        r, theta, phi = self.center_grd.coord_sys.ToSpherical(self.center_grd.grid.x,
                                                               self.center_grd.grid.y,
                                                               self.center_grd.grid.z)
        co,cx,crho = CO(theta,phi)
        E_co = dotproduct(E,co)
        E_cx = dotproduct(E,cx)
        power_beam = np.abs(E_co)**2 + np.abs(E_cx)**2 
        peak = power_beam.max()
        print('Gain',10*np.log10(peak))
        NN = np.where((np.abs(E_co)**2 + np.abs(E_cx)**2 )/peak > 10**(-20/10))[0]
        print(np.log10(np.abs(E_co).max()**2)*10,np.log10(np.abs(E_cx).max()**2)*10)

        r= polar_angle.polarization_angle(np.concatenate((E_co[NN],E_cx[NN])).reshape(2,-1))
        print('rotation angle method 3: ',r.x*180/np.pi-dAz, r.status)
        r2= polar_angle.polarization_angle_method2(np.concatenate((E_co[NN],E_cx[NN])).reshape(2,-1))
        print('rotation angle method 2: ',r2.x*180/np.pi-dAz, r2.status)
        r1= polar_angle.polarization_angle_method1(np.concatenate((E_co[NN],E_cx[NN])).reshape(2,-1))
        print('rotation angle method 1: ',r1.x*180/np.pi-dAz, r1.status)
        print(r.x*180/np.pi)
        Beam_new = polar_angle.rotation_angle(r.x,np.concatenate((E_co,E_cx)).reshape(2,-1))
        E_co_new = Beam_new[0,:]
        E_cx_new = Beam_new[1,:]

        if plot:
            Nx = x.size
            Ny = y.size
            vmax1 = np.abs(E_co).max()
            vmax2 = np.abs(E_cx).max()
            vmax = np.log10(max(vmax1, vmax2))*20
            vmax = np.log10(peak)*10
            fig, ax = plt.subplots(1,2, figsize=(12, 4.7))
            fig.suptitle('r =  '+str(dx)+'mm,'+' Rx Oritentation: ' + str(dAz)+'deg',fontsize = 13)
            p1 = ax[0].pcolor(x,y, 10*np.log10(np.abs(E_co.reshape(Ny,Nx))**2),vmax = vmax,vmin= vmax-80,cmap = cmap)
            ax[0].axis('equal')
            ax[0].set_title('co-polar beam',fontsize = 15)
            p2 = ax[1].pcolor(x,y, 10*np.log10(np.abs(E_cx.reshape(Ny,Nx))**2),vmax = vmax,vmin= vmax-80,cmap = cmap)
            ax[1].set_title('cx-polar beam',fontsize  = 15)
            ax[1].axis('equal')
            cbar = fig.colorbar(p1, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
            plt.savefig(picture_fname1, dpi=200)
            plt.show()

            '''
            fig, ax = plt.subplots(1,2, figsize=(12, 4.7))
            fig.suptitle('r =  '+str(dx)+'mm,'+' Rx Oritentation: ' + str(dAz)+'deg',fontsize = 13)
            p1 = ax[0].pcolor(x,y, np.angle(E_co.reshape(Ny,Nx)),cmap = cmap)
            ax[0].axis('equal')
            ax[0].set_title('co-polar beam',fontsize = 15)
            p2 = ax[1].pcolor(x,y, np.angle(E_cx.reshape(Ny,Nx)),cmap = cmap)
            ax[1].set_title('cx-polar beam',fontsize  = 15)
            ax[1].axis('equal')
            cbar = fig.colorbar(p1, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
            plt.savefig(picture_fname1, dpi=200)
            plt.show()
            '''
            
            vmax1 = np.abs(E_co_new).max()
            vmax2 = np.abs(E_cx_new).max()
            #vmax = np.log10(max(vmax1, vmax2))*20
            fig, ax = plt.subplots(1,2, figsize=(12, 4.7))
            fig.suptitle('Optimal Rotation Angle:'+str(r.x[0]*180/np.pi)+' deg'+', Beam after rotation', fontsize = 13)
            p3 = ax[0].pcolor(x,y, 10*np.log10(np.abs(E_co_new.reshape(Ny,Nx))**2),vmax = vmax,vmin= vmax-80,cmap = cmap)
            p4 = ax[1].pcolor(x,y, 10*np.log10(np.abs(E_cx_new.reshape(Ny,Nx))**2),vmax = vmax,vmin= vmax-80,cmap = cmap)
            cbar = fig.colorbar(p3, ax=[ax[0], ax[1]], orientation='vertical', fraction=0.05, pad=0.1)
            ax[1].axis('equal')
            ax[0].axis('equal')
            plt.savefig(picture_fname2, dpi=300)
            plt.show()

            '''
            fig, ax = plt.subplots(1,2, figsize=(12, 4.7))
            fig.suptitle('r =  '+str(dx)+'mm,'+' Rx Oritentation: ' + str(dAz)+'deg',fontsize = 13)
            p1 = ax[0].pcolor(x,y, np.angle(E_co_new.reshape(Ny,Nx)),cmap = cmap)
            ax[0].axis('equal')
            ax[0].set_title('co-polar beam',fontsize = 15)
            p2 = ax[1].pcolor(x,y, np.angle(E_cx_new.reshape(Ny,Nx)),cmap = cmap)
            ax[1].set_title('cx-polar beam',fontsize  = 15)
            ax[1].axis('equal')
            cbar = fig.colorbar(p1, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
            plt.savefig(picture_fname1, dpi=200)
            plt.show()
            '''

            
            ###############
            # plot Stoke parameter beams 1
            '''
            vmax = 0
            vmin = -40
            I, Q, U, V = stoke_param.stokes_beams(E_co,E_cx)
            fig, ax = plt.subplots(2,2, figsize=(12, 4.7*2))
            fig.suptitle('stoke beams',fontsize = 13)
            fig.suptitle('r =  '+str(dx)+'mm,'+' Rx Oritentation: ' + str(dAz)+'deg',fontsize = 13)
            p1 = ax[0,0].pcolor(x,y, 10*np.log10(np.abs(I.reshape(Ny,Nx))),vmax = vmax,vmin= vmin,cmap = cmap)
            ax[0,0].axis('equal')
            ax[0,0].set_title('Intensity beam',fontsize = 15)

            p2 = ax[0,1].pcolor(x,y, 10*np.log10(np.abs(V.reshape(Ny,Nx))),vmax = vmax,vmin= vmin,cmap = cmap)
            ax[0,1].set_title('V',fontsize  = 15)
            ax[0,1].axis('equal')

            p3 = ax[1,0].pcolor(x,y, 10*np.log10(np.abs(Q.reshape(Ny,Nx))),vmax = vmax,vmin= vmin,cmap = cmap)
            ax[1,0].axis('equal')
            ax[1,0].set_title('Q',fontsize = 15)

            p4 = ax[1,1].pcolor(x,y, 10*np.log10(np.abs(U.reshape(Ny,Nx))),vmax = vmax,vmin= vmin,cmap = cmap)
            ax[1,1].set_title('U',fontsize  = 15)
            ax[1,1].axis('equal')
            cbar = fig.colorbar(p1, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
            #plt.savefig(picture_fname1, dpi=200)
            plt.show()

            # plot Stoke parameter beams new
            I, Q, U, V = stoke_param.stokes_beams(E_co_new,E_cx_new)
            fig, ax = plt.subplots(2,2, figsize=(12, 4.7*2))
            fig.suptitle('stoke beams after rotation',fontsize = 13)
            p1 = ax[0,0].pcolor(x,y, 10*np.log10(np.abs(I.reshape(Ny,Nx))),vmax = vmax,vmin= vmin,cmap = cmap)
            ax[0,0].axis('equal')
            ax[0,0].set_title('Intensity beam',fontsize = 15)

            p2 = ax[0,1].pcolor(x,y, 10*np.log10(np.abs(V.reshape(Ny,Nx))),vmax = vmax,vmin= vmin,cmap = cmap)
            ax[0,1].set_title('V',fontsize  = 15)
            ax[0,1].axis('equal')

            p3 = ax[1,0].pcolor(x,y, 10*np.log10(np.abs(Q.reshape(Ny,Nx))),vmax = vmax,vmin= vmin,cmap = cmap)
            ax[1,0].axis('equal')
            ax[1,0].set_title('Q',fontsize = 15)

            p4 = ax[1,1].pcolor(x,y, 10*np.log10(np.abs(U.reshape(Ny,Nx))),vmax = vmax,vmin= vmin,cmap = cmap)
            ax[1,1].set_title('U',fontsize  = 15)
            ax[1,1].axis('equal')
            cbar = fig.colorbar(p1, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
            #plt.savefig(picture_fname1, dpi=200)
            plt.show()
            '''

            
        return r.x*180/np.pi-dAz
    def read_beam(self,field_name = None):
        if field_name == None:
            field_name = self.field_grid_fname
        x, y, Ex, Ey, Ez = field_storage.read_grd(field_name)
        E = vector()
        E.x = Ex
        E.y = Ey
        E.z = Ez

        r, theta, phi = self.center_grd.coord_sys.ToSpherical(self.center_grd.grid.x,
                                                               self.center_grd.grid.y,
                                                               self.center_grd.grid.z)
        co,cx,crho = CO(theta,phi)
        E_co = dotproduct(E,co)
        E_cx = dotproduct(E,cx)
        return E_co, E_cx,x,y

