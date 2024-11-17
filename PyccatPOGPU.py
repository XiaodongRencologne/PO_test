#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np;
import torch as T;
import copy ;
import time;

DEVICE0=T.device('cpu')
c=299792458;
'''
some germetrical parametrs
'''
Theta_0=0.927295218001612; # offset angle of MR;
Ls      = 12000.0;           # distance between focal point and SR
Lm      = 6000.0;            # distance between MR and SR;
L_fimag=18000+Ls;
F=20000;


# In[2]:

# import the Firchhoffpy package;
import Kirchhoffpy;
# the intergration funciton by using scalar diffraction theory;
#from Kirchhoffpy.Kirchhoff import Complex,PO_scalar;
from POpy1 import Complex;
from POpyGPU import PO_GPU as PO
from POpyGPU import PO_far;
from POpy1 import MATRIX,crossproduct,scalarproduct
# 1. define the guassian beam of the input feed;
from Kirchhoffpy.Feedpy import Gaussibeam;
# 2. translation between coordinates system;
from Kirchhoffpy.coordinate_operations import Coord;
from Kirchhoffpy.coordinate_operations import Transform_local2global as local2global;
from Kirchhoffpy.coordinate_operations import Transform_global2local as global2local;
from Kirchhoffpy.coordinate_operations import cartesian_to_spherical as cart2spher;
# 3. mirror
from Kirchhoffpy.mirrorpy import profile,squarepanel,deformation,ImagPlane,adjuster;
# 4. field in source region;
from Kirchhoffpy.Spheical_field import spheical_grid;
# 5. inference function;
from Kirchhoffpy.inference import DATA2CUDA,fitting_func;


'''
1. read the input parameters
'''
def read_input(inputfile):
    coefficient_m2=np.genfromtxt(inputfile+'/coeffi_m2.txt',delimiter=',');
    coefficient_m1=np.genfromtxt(inputfile+'/coeffi_m1.txt',delimiter=',');
    List_m2=np.genfromtxt(inputfile+'/List_m2.txt',delimiter=',');
    List_m1=np.genfromtxt(inputfile+'/List_m1.txt',delimiter=',');
    parameters=np.genfromtxt(inputfile+'/input.txt',delimiter=',')[...,1];
    electro_params=np.genfromtxt(inputfile+'/electrical_parameters.txt',delimiter=',')[...,1];
    
    M2_size=parameters[0:2];M1_size=parameters[2:4];
    R2=parameters[4];R1=parameters[5];
    p_m2=parameters[6];q_m2=parameters[7];
    p_m1=parameters[8];q_m1=parameters[9];
    M2_N=parameters[10:12];M1_N=parameters[12:14]
    fimag_N=parameters[14:16];fimag_size=parameters[16:18]
    distance=parameters[18];
    freq=electro_params[0]*10**9;
    edge_taper=electro_params[1];
    Angle_taper=electro_params[2]/180*np.pi;
    Lambda=c/freq*1000;
    k=2*np.pi/Lambda;
    
    return coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,distance,edge_taper,Angle_taper,k;
    

'''
2. produce the coordinates relationship;
'''
def relation_coorsys(Theta_0,Ls,Lm,L_fimag,F,defocus):
    '''
    germetrical parametrs
    
    Theta_0=0.927295218001612; # offset angle of MR;
    Ls      = 12000.0;           # distance between focal point and SR
    Lm      = 6000.0;            # distance between MR and SR;
    L_fimag=18000+Ls;
    F=20000;
    #defocus# is the defocus of receiver;
    '''
    
    '''
    #angle# is angle change of local coordinates and global coordinates;
    #D#     is the distance between origin of local coord and global coord in global coordinates;
    '''

    angle_m2=[-(np.pi/2+Theta_0)/2,0,0] #  1. m2 and global co-ordinates
    D_m2=[0,-Lm*np.sin(Theta_0),0]
    
    angle_m1=[-Theta_0/2,0,0]          #  2. m1 and global co-ordinates
    D_m1=[0,0,Lm*np.cos(Theta_0)]
    
    angle_s=[0,np.pi,0];               #  3. source and global co-ordinates
    D_s=[0,0,0];
    
    angle_fimag=[-Theta_0,0,0];        #  4. fimag and global co-ordinates
    defocus_fimag=[0,0,0];
    defocus_fimag[2]=1/(1/F-1/(Ls+defocus[2]))+L_fimag;
    defocus_fimag[1]=(F+L_fimag-defocus_fimag[2])/F*defocus[1];
    defocus_fimag[0]=(F+L_fimag-defocus_fimag[2])/F*defocus[0];
    D_fimag=[0,0,0]
    D_fimag[0]=defocus_fimag[0];
    D_fimag[1]=defocus_fimag[1]*np.cos(Theta_0)-np.sin(Theta_0)*(L_fimag-defocus_fimag[2]+Lm);
    D_fimag[2]=-defocus_fimag[1]*np.sin(Theta_0)-np.cos(Theta_0)*(L_fimag-defocus_fimag[2]);
    
    # 5. feed and global co-ordinate
    '''
    C=1/(1/Lm-1/F)+defocus[2]+Ls;
    C=21000;
    angle_f=[np.pi/2-defocus[1]/C,0,-defocus[0]/C]; 
    D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]];
    '''
    angle_f=[np.pi/2,0,0];    
    D_f=[defocus[0],Ls+defocus[2]-Lm*np.sin(Theta_0),-defocus[1]]
    
    
    return angle_m2,D_m2,angle_m1,D_m1,angle_fimag,D_fimag,angle_f,D_f,angle_s,D_s;

'''
3.  build the CCAT-P model
'''
# build the model for ccat-prime and imaginary plane;
def model_ccat(coefficient_m2,List_m2,M2_sizex,M2_sizey,M2_Nx,M2_Ny,R2,# m2
          coefficient_m1,List_m1,M1_sizex,M1_sizey,M1_Nx,M1_Ny,R1, # m1
          Rangex,Rangey,fimag_Nx,fimag_Ny,# imaginary field
          ad_m2,ad_m1,p_m2,q_m2,p_m1,q_m1):              #fimag & initial position of adjusters;
    
    surface_m2=profile(coefficient_m2,R2);# define the surface function of m2;
    surface_m1=profile(coefficient_m1,R1);# define the surface function of m1;    
    m2,m2_n,m2_dA=squarepanel(List_m2[...,0],List_m2[...,1],M2_sizex,M2_sizey,M2_Nx,M2_Ny,surface_m2);
    m1,m1_n,m1_dA=squarepanel(List_m1[...,0],List_m1[...,1],M1_sizex,M1_sizey,M1_Nx,M1_Ny,surface_m1);
    fimag,fimag_n,fimag_dA=ImagPlane(Rangex,Rangey,fimag_Nx,fimag_Ny);
    
    # modified the panel based on the initial adjusters distribution;
    Panel_N_m2=int(List_m2.size/2)
    
    m2_dz=deformation(ad_m2.ravel(),List_m2,p_m2,q_m2,m2);
    m1_dz=deformation(ad_m1.ravel(),List_m1,p_m1,q_m1,m1);
    
    m2.z=m2.z+m2_dz;
    m1.z=m1.z-m1_dz;

    return m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA;     



'''
4. the function gives the calculation orders;
'''
def First_computing_far(m2,m2_n,m2_dA, # Mirror 2,
                    m1,m1_n,m1_dA,# Mirror 1,
                    fimag,fimag_n,fimag_dA,defocus, # imaginary focal plane,
                    source,   # source
                    k,Theta_max,E_taper,Keepmatrix=False): # frequency and edge taper;
    
    start=time.perf_counter();

    angle_m2,D_m2,angle_m1,D_m1,angle_fimag,D_fimag,angle_f,D_f,angle_s,D_s=relation_coorsys(Theta_0,Ls,Lm,L_fimag,F,defocus);
    '''
    1. get the field on m2;
    '''
    # get the field on m2 and incident angle in feed coordinates;
    
    m2=local2global(angle_m2,D_m2,m2);
    m2_n=local2global(angle_m2,[0,0,0],m2_n);
    Field_m2=Complex(); # return 2

    Field_m2_E,Field_m2_H=Gaussibeam(E_taper,Theta_max,k,m2,m2_n,angle_f,D_f,polarization='x');
    
    Field_m2_E.N=[];Field_m2_H.N=[];
    Field_m2_E=local2global(angle_f,[0,0,0],Field_m2_E)
    Field_m2_H=local2global(angle_f,[0,0,0],Field_m2_H)
    '''
    2. calculate the field on imaginary focal plane;
    '''
    fimag=local2global(angle_fimag,D_fimag,fimag);
    fimag_n=local2global(angle_fimag,[0,0,0],fimag_n);
    
    
    m2_n.x=-m2_n.x;m2_n.y=-m2_n.y;m2_n.z=-m2_n.z;
    Field_m2_E=scalarproduct(2,Field_m2_E);
    Field_m2_H=scalarproduct(2,Field_m2_H);
    Field_fimag_E,Field_fimag_H=PO(m2,m2_n,m2_dA,fimag,0,Field_m2_H,-k);
    
    
    print('2')
    
    '''
    3. calculate the field on m1;
    '''
    #print('3')
    m1=local2global(angle_m1,D_m1,m1);
    m1_n=local2global(angle_m1,[0,0,0],m1_n);
    
    Field_fimag_E=scalarproduct(1,Field_fimag_E);
    Field_fimag_H=scalarproduct(1,Field_fimag_H);
    Field_m1_E,Field_m1_H=PO(fimag,fimag_n,fimag_dA,m1,Field_fimag_E,Field_fimag_H,k);

    print('3')
    '''
    4. calculate the field in source;
    '''
    #print('4')
    source=local2global(angle_s,D_s,source);
    Field_s_E,Field_s_H=PO_far(m1,m1_n,m1_dA,source,0,Field_m1_H,k);

    print('4')
    elapsed =(time.perf_counter()-start);
    print('time used:',elapsed);
    
    return Field_fimag_E,Field_fimag_H,Field_m1_E,Field_m1_H,Field_s_E,Field_s_H;
'''
4. the function gives the calculation orders;
'''
def First_computing(m2,m2_n,m2_dA, # Mirror 2,
                    m1,m1_n,m1_dA,# Mirror 1,
                    fimag,fimag_n,fimag_dA,defocus, # imaginary focal plane,
                    source,   # source
                    k,Theta_max,E_taper,Keepmatrix=False): # frequency and edge taper;
    
    start=time.perf_counter();

    angle_m2,D_m2,angle_m1,D_m1,angle_fimag,D_fimag,angle_f,D_f,angle_s,D_s=relation_coorsys(Theta_0,Ls,Lm,L_fimag,F,defocus);
    '''
    1. get the field on m2;
    '''
    # get the field on m2 and incident angle in feed coordinates;
    
    m2=local2global(angle_m2,D_m2,m2);
    m2_n=local2global(angle_m2,[0,0,0],m2_n);
    Field_m2=Complex(); # return 2

    Field_m2_E,Field_m2_H=Gaussibeam(E_taper,Theta_max,k,m2,m2_n,angle_f,D_f,polarization='x');
    
    Field_m2_E.N=[];Field_m2_H.N=[];
    Field_m2_E=local2global(angle_f,[0,0,0],Field_m2_E)
    Field_m2_H=local2global(angle_f,[0,0,0],Field_m2_H)
    '''
    2. calculate the field on imaginary focal plane;
    '''
    fimag=local2global(angle_fimag,D_fimag,fimag);
    fimag_n=local2global(angle_fimag,[0,0,0],fimag_n);
    
    
    m2_n.x=-m2_n.x;m2_n.y=-m2_n.y;m2_n.z=-m2_n.z;
    Field_m2_E=scalarproduct(2,Field_m2_E);
    Field_m2_H=scalarproduct(2,Field_m2_H);
    Field_fimag_E,Field_fimag_H=PO(m2,m2_n,m2_dA,fimag,0,Field_m2_H,-k);
    
    
    print('2')
    
    '''
    3. calculate the field on m1;
    '''
    #print('3')
    m1=local2global(angle_m1,D_m1,m1);
    m1_n=local2global(angle_m1,[0,0,0],m1_n);
    
    Field_fimag_E=scalarproduct(1,Field_fimag_E);
    Field_fimag_H=scalarproduct(1,Field_fimag_H);
    Field_m1_E,Field_m1_H=PO(fimag,fimag_n,fimag_dA,m1,Field_fimag_E,Field_fimag_H,k);

    print('3')
    '''
    4. calculate the field in source;
    '''
    #print('4')
    source=local2global(angle_s,D_s,source);
    Field_m1_E.x=np.array([0.0+1j*0.0])
    Field_m1_E.y=np.array([0.0+1j*0.0])
    Field_m1_E.z=np.array([0.0+1j*0.0])
    Field_s_E,Field_s_H=PO(m1,m1_n,m1_dA,source,0,Field_m1_H,k);

    print('4')
    elapsed =(time.perf_counter()-start);
    print('time used:',elapsed);
    
    return Field_fimag_E,Field_fimag_H,Field_m1_E,Field_m1_H,Field_s_E,Field_s_H;

'''
5. the function is the used to calculate the field
'''
def field_calculation(inputfile,source_field,defocus,ad_m2,ad_m1):
    
    # 0. read the input parameters from the input files;
    coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,distance,edge_taper,Angle_taper,k=read_input(inputfile);
    
    # 1. produce the coordinate system;
    # 2. build model;
    m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA=model_ccat(coefficient_m2,List_m2,M2_size[0],M2_size[1],M2_N[0],M2_N[1],R2,
                                                                  coefficient_m1,List_m1,M1_size[0],M1_size[1],M1_N[0],M1_N[1],R1,
                                                                  fimag_size[0],fimag_size[1],fimag_N[0],fimag_N[1],
                                                                  ad_m2,ad_m1,p_m2,q_m2,p_m1,q_m1);
    
    # 3.calculate the source beam
    Field_fimag_E,Field_fimag_H,Field_m1_E,Field_m1_H,Field_s_E,Field_s_H=First_computing(m2,m2_n,m2_dA,
                                                                                          m1,m1_n,m1_dA,
                                                                                          fimag,fimag_n,
                                                                                          fimag_dA,defocus,
                                                                                          source_field,k,
                                                                                          Angle_taper,edge_taper,Keepmatrix=False);
    
    return Field_fimag_E,Field_fimag_H,Field_m1_E,Field_m1_H,Field_s_E,Field_s_H;
'''
5.2 the function is the used to calculate the far field
'''
def field_calculation_far(inputfile,source_field,defocus,ad_m2,ad_m1):
    
    # 0. read the input parameters from the input files;
    coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,distance,edge_taper,Angle_taper,k=read_input(inputfile);
    
    # 1. produce the coordinate system;
    # 2. build model;
    m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA=model_ccat(coefficient_m2,List_m2,M2_size[0],M2_size[1],M2_N[0],M2_N[1],R2,
                                                                  coefficient_m1,List_m1,M1_size[0],M1_size[1],M1_N[0],M1_N[1],R1,
                                                                  fimag_size[0],fimag_size[1],fimag_N[0],fimag_N[1],
                                                                  ad_m2,ad_m1,p_m2,q_m2,p_m1,q_m1);
    
    # 3.calculate the source beam
    Field_fimag_E,Field_fimag_H,Field_m1_E,Field_m1_H,Field_s_E,Field_s_H=First_computing_far(m2,m2_n,m2_dA,
                                                                                              m1,m1_n,m1_dA,
                                                                                              fimag,fimag_n,
                                                                                              fimag_dA,defocus,
                                                                                              source_field,
                                                                                              k,Angle_taper,
                                                                                              edge_taper,Keepmatrix=False)
    
    return Field_fimag_E,Field_fimag_H,Field_m1_E,Field_m1_H,Field_s_E,Field_s_H;
    



    
