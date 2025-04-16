#from tqdm import tqdm
import numpy as np
import torch as T
from numba import njit, prange
from .Vopy import vector,crossproduct,scalarproduct,abs_v,dotproduct,sumvector,abs_v_Field

from .POpyGPU import PO_GPU_2 as PO_GPU
from .POpyGPU import PO_far_GPU2 as PO_far_GPU

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
           face2,face2_n,#face2_dS,
           #face3,
           Field_in_E,Field_in_H,k,n,device =T.device('cuda')):
    n0 = 1
    k_n = k*n
    Z = Z0/n
    
    # calculate the transmission and reflection on face 1.
    f1_E_t,f1_E_r,f1_H_t,f1_H_r, p_n1 , T1, R1, NN = calculate_Field_T_R(n0,n,face1_n,Field_in_E,Field_in_H)
    print('output poynting:')
    p_t_n1 = poyntingVector(f1_E_t,f1_H_t)
    print(abs_v(p_t_n1).max())
    start_time = time.time()
    F2_in_E,F2_in_H = PO_GPU(face1,face1_n,face1_dS,
                           face2,
                           f1_E_t,f1_H_t,
                           k,n,
                           device = device)
    print(time.time() - start_time)
    f2_E_t,f2_E_r,f2_H_t,f2_H_r, p_n2, T2, R2, NN= calculate_Field_T_R(n,n0,face2_n,F2_in_E,F2_in_H)
    print('output poynting:')
    p_t_n2 = poyntingVector(f2_E_t,f2_H_t)
    #p_t_n1 = scalarproduct(1/abs_v(p_t_n1),p_t_n1)
    print(abs_v(p_t_n2).max())
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
    b = vector()
    b.x = np.conjugate(B.x)
    b.y = np.conjugate(B.y)
    b.z = np.conjugate(B.z)
    
    C= crossproduct(A,b)
    C.x = C.x.real
    C.y = C.y.real
    C.z = C.z.real
    return C


def Fresnel_coeffi(n1,n2,theta_i_cos):
    # 4. calculate the transmission and reflection coefficient
    # calculate the angle of refraction
    theta_i_sin = np.sqrt(1 - theta_i_cos**2)
    theta_t_sin = n1/n2*theta_i_sin
    theta_t_cos = np.sqrt(1 - theta_t_sin**2)
    t_p = 2*n1*theta_i_cos/(n2 * theta_i_cos + n1 * theta_t_cos)
    t_s = 2*n1*theta_i_cos/(n1 * theta_i_cos + n2 * theta_t_cos)

    r_p = (n2*theta_i_cos - n1*theta_t_cos)/(n2*theta_i_cos + n1*theta_t_cos)
    r_s = (n1*theta_i_cos - n2*theta_t_cos)/(n1*theta_i_cos + n2*theta_t_cos)
    '''
    print('check the Fresnel coefficient')
    print(np.abs(r_s**2 + n2*theta_t_cos/theta_i_cos*t_s**2 - 1).max())
    print(np.abs(r_s**2 + n2*theta_t_cos/theta_i_cos*t_s**2 - 1).min())
    print(np.abs(r_p**2 + n2*theta_t_cos/theta_i_cos*t_p**2 - 1).max())
    print(np.abs(r_p**2 + n2*theta_t_cos/theta_i_cos*t_p**2 - 1).min())
    '''
    return t_p,t_s,r_p,r_s

def calculate_Field_T_R(n1,n2,v_n,E,H):
    # calculate poynting vector,
    # here assuming wave vector k has same direction with poynting vector.
    poynting_i = poyntingVector(E,H)
    poynting_i_A = abs_v(poynting_i)
    k_i = scalarproduct(1/poynting_i_A,poynting_i)

    # 1. incident angle
    theta_i_cos = dotproduct(v_n,k_i)
    theta_i = np.arccos(theta_i_cos)
    if np.sum(theta_i_cos > 0) < np.sum(theta_i_cos < 0):
        print('#$%^&&*&*())_')
        v_n = scalarproduct(-1,v_n)
        theta_i_cos = np.abs(theta_i_cos)
        theta_i = np.arccos(theta_i_cos)
    else:
        pass
    
    theta_i_sin = np.sqrt(1 - theta_i_cos**2)
    theta_t_sin = n1/n2*theta_i_sin
    theta_t_cos = np.sqrt(1 - theta_t_sin**2)
    theta_t = np.arccos(theta_t_cos)

    # 1.2 wave vector k_i, k_r, k_t
    k_r = sumvector(k_i,scalarproduct(-2*dotproduct(k_i,v_n),v_n))
    k_i_n = scalarproduct(-theta_i_cos,v_n)
    k_i_p = sumvector(k_i,k_i_n)
    k_t = sumvector(scalarproduct(n1/n2,k_i_p),
                    scalarproduct(theta_t_cos,v_n))


    # 2. calculate the vector s that is perpendicular to the plane of incidence.
    s = crossproduct(k_i,v_n)
    s_A = abs_v(s)#/(abs_v(v_n)*abs_v(k_i))
    print('check the sin(theta_i)')
    #print(s_A.reshape(11,11))
    threshold = 10**(-18)
    NN = np.where(s_A <= threshold)
    if NN[0].size != 0:
        print('weird data!!!!!!!')
        ref_vector = np.array([1.0,0.0,0.0],dtype = np.float64)
        ref_vector2 = np.array([0.0,1.0,0.0],dtype = np.float64)
        for i in NN[0]:
            new_s = np.cross(np.array([v_n.x[i],v_n.y[i],v_n.z[i]]),ref_vector)
            if np.allclose(new_s,0):
                new_s = np.cross(np.array([v_n.x[i],v_n.y[i],v_n.z[i]]),ref_vector2)
            s.x[i] = new_s[0]
            s.y[i] = new_s[1]
            s.z[i] = new_s[2]
            s_A[i] = np.sqrt(s.x[i]**2+s.y[i]**2+s.z[i]**2)
    s = scalarproduct(1/s_A,s)
    # 3. get the third vector
    x_n = crossproduct(s,v_n)
    x_n = scalarproduct(1/abs_v(x_n),x_n)
    # get parallel vector p_r and p_t
    p_r = crossproduct(k_r,s)
    p_t = crossproduct(k_t,s)
    p_i = crossproduct(k_i,s)

    # 4. calculate the transmission and reflection coefficient
    t_p,t_s,r_p,r_s = Fresnel_coeffi(n1,n2,theta_i_cos)

    # 5. convert E and H field to s_n and p_n, perpendicutlar and paraller
    E_s = scalarproduct(dotproduct(E,s),s)
    E_p = scalarproduct(dotproduct(E,p_i),p_i)
    #E_p = sumvector(E,scalarproduct(-1,E_s))
    E_p_z = scalarproduct(dotproduct(E,v_n),v_n)
    E_p_x = scalarproduct(dotproduct(E,x_n),x_n)
    #print('check v_n, x_n, s')
    #printF(sumvector(crossproduct(v_n,x_n),scalarproduct(-1,s)))
    #printF(sumvector(crossproduct(s,v_n),scalarproduct(-1,x_n)))
    #printF(sumvector(crossproduct(x_n,s),scalarproduct(-1,v_n)))

    H_s = scalarproduct(dotproduct(H,s),s)
    #H_p = sumvector(H,scalarproduct(-1,H_s))
    H_p = scalarproduct(dotproduct(H,p_i),p_i)
    #'''
    # 6. calculate the transmission and reflection field
    E_t_s = scalarproduct(t_s,E_s)
    E_t_p_z = scalarproduct(t_p*n1/n2,E_p_z)#
    E_t_p_x = scalarproduct(t_p*theta_t_cos/theta_i_cos,E_p_x)#
    E_t = sumvector(E_t_s,sumvector(E_t_p_x,E_t_p_z))
    #E_t = sumvector(E_t_s,E_t_p_x)

    E_r_s = scalarproduct(r_s,E_s)
    E_r_p_z = scalarproduct(r_p,E_p_z)#
    E_r_p_x = scalarproduct(-r_p,E_p_x)#
    E_r = sumvector(E_r_s,sumvector(E_r_p_x,E_r_p_z))
    #E_r = sumvector(E_r_s,E_r_p_x)
    '''
    # 6. calculate the transmission and reflection field
    E_t_s = scalarproduct(t_s,E_s)
    E_t_p = scalarproduct(t_p, scalarproduct(dotproduct(E,p_t),p_t))
    E_t = sumvector(E_t_s,E_t_p)

    E_r_s = scalarproduct(r_s,E_s)
    E_r_p = scalarproduct(r_p,scalarproduct(dotproduct(E,p_r),p_r))
    E_r = sumvector(E_r_s,E_r_p)
    '''

    # get H-field 
    H_r = scalarproduct(n1,crossproduct(k_r,E_r))
    H_t = scalarproduct(n2,crossproduct(k_t,E_t))

    print('##############')
    poynting_t = poyntingVector(E_t,H_t)
    poynting_t_A = abs_v(poynting_t)
    poynting_r = poyntingVector(E_r,H_r)
    poynting_r_A = abs_v(poynting_r)
    print('check energy conservation!')
    print('check the poynting vector')  
    print(poynting_i_A.max(),poynting_i_A.min())
    #error = np.abs(poynting_t_A*theta_t_cos + poynting_r_A*theta_i_cos - theta_i_cos*poynting_i_A).reshape(11,11)
    
    #N=3
    #print(N)
    #print(E.x.reshape(11,11)[5,N])
    #print(E_r.x.reshape(11,11)[5,N])
    #print(E_t.x.reshape(11,11)[5,N])  
    #print(E.y.reshape(11,11)[5,N])
    #print(E_r.y.reshape(11,11)[5,N])
    #print(E_t.y.reshape(11,11)[5,N])
    #print(E.z.reshape(11,11)[5,N])
    #print(E_r.z.reshape(11,11)[5,N])
    #print(E_t.z.reshape(11,11)[5,N])

    #N=[0,1,2,3,4,5]
    #print(N)
    #print(E.x.reshape(11,11)[5,N])
    #print(E_r.x.reshape(11,11)[5,N])
    #print(E_t.x.reshape(11,11)[5,N])   
    #print(np.abs(E.x + E_r.x - E_t.x).reshape(11,11)[N,5])
    #print(np.abs(E.y + E_r.y - E_t.y).reshape(11,11)[N,5])
    #print(np.abs(E.z + E_r.z - n2*E_t.z).reshape(11,11)[N,5])
    #NN = np.where(error == error.max())
    #print(NN)
    #print(error[NN])
    #print(poynting_i_A.reshape(11,11)[NN])
    
    print('check boundary conditions!!')
    #print(E.x + E_r.x - E_t.x)
    print('E field:')
    #print((np.abs(Field_in_E.x + f1_E_r.x - f1_E_t.x)/np.abs(Field_in_E.x)))
    #print(E.x.reshape(11,11)[5,5])
    #print(E_r.x.reshape(11,11)[5,5])
    #print(E_t.x.reshape(11,11)[5,5])
    
    return E_t,E_r,H_t,H_r, poynting_i, n2/n1*theta_t_cos/theta_i_cos*(t_p**2+t_s**2)/2, (r_p**2+r_s**2)/2,NN

