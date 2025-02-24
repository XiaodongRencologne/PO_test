#from tqdm import tqdm
import numpy as np;
import torch as T;
from numba import njit, prange;
from .Vopy import vector,crossproduct,scalarproduct,abs_v,dotproduct,sumvector

from .POpyGPU import PO_GPU, PO_far_GPU

import copy;
import time;
c=299792458;
mu=4*np.pi*10**(-7);
epsilon=8.854187817*10**(-12);
Z0=np.sqrt(mu/epsilon,dtype = np.float64)

def printF(f):
    print('x')
    print(f.x.reshape(101,-1))#[49:52,:])
    print('y')
    print(f.y.reshape(101,-1))#[49:52,:])
    print('z')
    print(f.z.reshape(101,-1))#[49:52,:])

'''testing'''
def lensPO(face1,face1_n,face1_dS,
           face2,face2_n,#face2_dS,
           #face3,
           Field_in_E,Field_in_H,k,n,device =T.device('cuda')):
    n0 = 1
    k_n = k*n
    
    # calculate the transmission and reflection on face 1.
    #print('H')
    #printF(Field_in_H)
    #print('E')
    #printF(Field_in_E)
    f1_E_t,f1_E_r,f1_H_t,f1_H_r, p_n1= Fresnel_coeffi(n0,n,face1_n,Field_in_E,Field_in_H)

    """
    print(p_n1.x)
    print(p_n1.y)
    N = 101
    N_0 = int(N/2)
    print(p_n1.z.reshape(N,-1)[N_0-5:N_0+6,N_0-5:N_0+6])
    print(face1_n.z.reshape(N,-1)[N_0-5:N_0+6,N_0-5:N_0+6])
    """
    
    F2_in_E,F2_in_H = PO_GPU(face1,face1_n,face1_dS,
                           face2,
                           f1_E_t,f1_H_t,
                           k_n,
                           device = device)
    
    f2_E_t,f2_E_r,f2_H_t,f2_H_r, p_n2= Fresnel_coeffi(n,n0,face2_n,F2_in_E,F2_in_H)
    
    printF(p_n2)

    """
    print(p_n2.x)
    print(p_n2.y)
    N = 101
    N_0 = int(N/2)
    print(p_n2.z.reshape(N,-1)[N_0-5:N_0+6,N_0-5:N_0+6])
    print(face2_n.z.reshape(N,-1)[N_0-5:N_0+6,N_0-5:N_0+6])

    """
    f2_E_t.x = np.nan_to_num(f2_E_t.x)
    f2_E_t.y = np.nan_to_num(f2_E_t.y)
    f2_E_t.z = np.nan_to_num(f2_E_t.z)
    
    f2_E_r.x = np.nan_to_num(f2_E_r.x)
    f2_E_r.y = np.nan_to_num(f2_E_r.y)
    f2_E_r.z = np.nan_to_num(f2_E_r.z)
    
    f2_H_t.x = np.nan_to_num(f2_H_t.x)
    f2_H_t.y = np.nan_to_num(f2_H_t.y)
    f2_H_t.z = np.nan_to_num(f2_H_t.z)
    
    f2_H_r.x = np.nan_to_num(f2_H_r.x)
    f2_H_r.y = np.nan_to_num(f2_H_r.y)
    f2_H_r.z = np.nan_to_num(f2_H_r.z)
    '''
    F_E,F_H = PO_GPU(face2,face2_n,face2_dS,
                     face3,
                     f2_E_t,f2_H_t,
                     k,
                     device = device)
    '''
    return F2_in_E,F2_in_H,f2_E_t,f2_E_r,f2_H_t,f2_H_r, f1_E_t,f1_E_r,f1_H_t,f1_H_r

def lensPO_far(face1,face1_n,face1_dS,
           face2,face2_n,face2_dS,
           face3,
           Field_in_E,Field_in_H,k,n,n0,device =T.device('cuda')):
    k_n = k*n
    # calculate the transmission and reflection on face 1.
    f1_E_t,f1_E_r,f1_H_t,f1_H_r = Fresnel_coeffi(n0,n,face1_n,Field_in_E,Field_in_H)

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

def poyntingVector(A,B):
    '''
    a = vector()
    a.x = np.abs(A.x)
    a.y = np.abs(A.y)
    a.z = np.abs(A.z)

    b = vector()
    b.x = np.abs(B.x)
    b.y = np.abs(B.y)
    b.z = np.abs(B.z)
    C= crossproduct(a,b)
    '''
    #A = abs_v(C)
    C= crossproduct(A,B)
    C.x = C.x.real
    C.y = C.y.real
    C.z = C.z.real
    return C

def Fresnel_coeffi(n1,n2,v_n,E,H):
    Z1 = Z0/n1
    Z2 = Z0/n2
    '''
    n1 and n2 re refractive index of material on left of the surface and right of the surface.
    v_n is the normal vector direction from n2 to n1
    poynting_n is the incident wave.
    theta_i = pi - arccos(v_n * poynting_n)
    '''
    #calculating poynting vector
    poynting_n = poyntingVector(E,H)
    A_poynting_n = abs_v(poynting_n)
    poynting_n = scalarproduct(1/A_poynting_n,poynting_n)
    
    # calculation the incident angle and refractive angle
    theta_i_cos = np.abs(dotproduct(v_n,poynting_n))
    theta_i_sin = np.sqrt(1 - theta_i_cos**2)
    theta_t_sin = n1/n2*theta_i_sin
    print(theta_t_sin)
    theta_t_cos = np.sqrt(1 - theta_t_sin**2)
    # define perpendicular vector, 
    #the plane give by normal vector v_n and poynting vector is the reflection and refractive plane. 
    # cross product of the two vector gives the vector perpendicular the reflection plane. We will use 
    # this vector as the reference to calculate the transmission coefficient for parallel polarization 
    # and perpendicular polarization coefficient.
    #s_n = scalarproduct(1/A_poynting_n, crossproduct(v_n,poynting_n))
    s_n = crossproduct(v_n,poynting_n)
    s_n = scalarproduct(1/abs_v(s_n),s_n)
    #print('s_n')
    #printF(s_n)
    nan_items = np.isnan(s_n.x)
    s_n.x[nan_items] = 0.0
    s_n.y[nan_items] = 0.0
    s_n.z[nan_items] = 0.0
    
    
    a = theta_i_cos
    d = theta_t_cos
    
    T_p = 2*n1*a/(n2 * a + n1 * d)
    T_s = 2*n1*a/(n1 * a + n2 * d)

    R_p = (n2*a - n1*d)/(n2*a + n1*d)
    R_s = (n1*a - n2*d)/(n1*a + n2*d) 

    #print(n2/n1*theta_t_cos/theta_i_cos*T_p**2 + R_p**2) 
    #print(n2/n1*theta_t_cos/theta_i_cos*T_s**2 + R_s**2)

    E_s = scalarproduct(dotproduct(E,s_n),s_n)
    E_p = sumvector(E,scalarproduct(-1,E_s)) 
    
    H_p = scalarproduct(dotproduct(H,s_n),s_n)
    H_s = sumvector(H,scalarproduct(-1,H_p))
    p_s = crossproduct(E_s,H_s)
    p_p = crossproduct(E_p,H_p)
    #printF(p_s)
    #printF(p_p)
    
    E_t = sumvector(scalarproduct(T_s,E_s),scalarproduct(T_p,E_p))
    E_r = sumvector(scalarproduct(R_s,E_s),scalarproduct(R_p,E_p))
    #print((theta_t_cos*abs_v(E_t)**2/Z2 + theta_i_cos*abs_v(E_r)**2/Z1)/theta_i_cos)
    
    H_t = sumvector(scalarproduct(T_s,H_s),scalarproduct(T_p,H_p))
    H_r = sumvector(scalarproduct(R_s,H_s),scalarproduct(R_p,H_p))
    

    
    return E_t,E_r,H_t,H_r, poynting_n
