#from tqdm import tqdm
import numpy as np
#import h5py
#from scipy.interpolate import CubicSpline
import torch as T
#from numba import njit, prange
#from .Vopy import vector,crossproduct,scalarproduct,abs_v,dotproduct,sumvector,abs_v_Field

from .POpyGPU import PO_GPU_2 as PO_GPU
from .POpyGPU import PO_far_GPU2 as PO_far_GPU

from .FresnelCoeff import poyntingVector,Fresnel_coeffi,calculate_Field_T_R, read_Fresnel_coeffi_AR, calculate_Field_T_R_AR
import copy
import time
c=299792458
mu=4*np.pi*10**(-7)
epsilon=8.854187817*10**(-12)
Z0=np.sqrt(mu/epsilon,dtype = np.float64)

def printF(f):
    N =int(np.sqrt(f.x.size))
    print('x')
    print(f.x)
    print('y')
    print(f.y)
    print('z')
    print(f.z)

'''testing'''
def lensPO(face1,face1_n,face1_dS,
           face2,face2_n,
           Field_in_E,Field_in_H,
           k,n,
           device =T.device('cuda')):
    n0 = 1
    k_n = k*n
    Z = Z0/n
    
    # calculate the transmission and reflection on face 1.
    f1_E_t,f1_E_r,f1_H_t,f1_H_r, p_n1 , T1, R1, NN = calculate_Field_T_R(n0,n,face1_n,Field_in_E,Field_in_H)
    #print('output poynting:')
    #p_t_n1 = poyntingVector(f1_E_t,f1_H_t)
    #print(abs_v(p_t_n1).max())
    start_time = time.time()
    F2_in_E,F2_in_H = PO_GPU(face1,face1_n,face1_dS,
                           face2,
                           f1_E_t,f1_H_t,
                           k,n,
                           device = device)
    #print(time.time() - start_time)
    f2_E_t,f2_E_r,f2_H_t,f2_H_r, p_n2, T2, R2, NN= calculate_Field_T_R(n,n0,face2_n,F2_in_E,F2_in_H)
    #print('output poynting:')
    p_t_n2 = poyntingVector(f2_E_t,f2_H_t)
    #p_t_n1 = scalarproduct(1/abs_v(p_t_n1),p_t_n1)
    #print(abs_v(p_t_n2).max())
    #printF(p_n2)
    
    return F2_in_E,F2_in_H,f2_E_t,f2_E_r,f2_H_t,f2_H_r, f1_E_t,f1_E_r,f1_H_t,f1_H_r,T1,R1,T2,R2

def lensPO_far(face1,face1_n,face1_dS,
           face2,face2_n,face2_dS,
           face3,
           Field_in_E,Field_in_H,k,n,n0,device =T.device('cuda')):
    k_n = k*n
    # calculate the transmission and reflection on face 1.
    f1_E_t,f1_E_r,f1_H_t,f1_H_r = calculate_Field_T_R(n0,n,face1_n,Field_in_E,Field_in_H)

    F2_in_E,F2_in_H = PO_GPU(face1,face1_n,face1_dS,
                           face2,
                           f1_E_t,f1_H_t,
                           k_n,
                           device = device)
    
    f2_E_t,f2_E_r,f2_H_t,f2_H_r = Fresnel_coeffi(n,n0,face1_n,F2_in_E,F2_in_H)

    F_E,F_H = PO_far_GPU(face2,face2_n,face2_dS,
                     face3,
                     f2_E_t,f2_H_t,
                     k,
                     device = device)
    return F_E,F_H


def lensPO_AR(face1,face1_n,face1_dS,
            face2,face2_n,
            Field_in_E,Field_in_H,
            k,n,
            AR_filename,
            groupname,
            device =T.device('cuda')):
    n0 = 1
    k_n = k*n
    Z = Z0/n
    AR1, AR2 = read_Fresnel_coeffi_AR(AR_filename, groupname, n0, n)
    # calculate the transmission and reflection on face 1.
    f1_E_t,f1_E_r,f1_H_t,f1_H_r, p_n1 , T1, R1, NN = calculate_Field_T_R_AR(n0,n,face1_n,Field_in_E,Field_in_H,AR1)
    #print('output poynting:')
    p_t_n1 = poyntingVector(f1_E_t,f1_H_t)
    #print(abs_v(p_t_n1).max())
    start_time = time.time()
    F2_in_E,F2_in_H = PO_GPU(face1,face1_n,face1_dS,
                           face2,
                           f1_E_t,f1_H_t,
                           k,n,
                           device = device)
    #print(time.time() - start_time)
    f2_E_t,f2_E_r,f2_H_t,f2_H_r, p_n2, T2, R2, NN= calculate_Field_T_R_AR(n,n0,face2_n,F2_in_E,F2_in_H,AR2)
    #print('output poynting:')
    p_t_n2 = poyntingVector(f2_E_t,f2_H_t)
    #p_t_n1 = scalarproduct(1/abs_v(p_t_n1),p_t_n1)
    #print(abs_v(p_t_n2).max())
    #printF(p_n2)
    
    return F2_in_E,F2_in_H,f2_E_t,f2_E_r,f2_H_t,f2_H_r, f1_E_t,f1_E_r,f1_H_t,f1_H_r,T1,R1,T2,R2


