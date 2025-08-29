import numpy as np
import h5py
from scipy.interpolate import CubicSpline

from .Vopy import vector,crossproduct,scalarproduct,abs_v,dotproduct,sumvector,abs_v_Field

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
    NN_r = np.where(np.abs(theta_t_sin)>=1.0) # total reflection point
    theta_t_sin[NN_r] =1.0
    theta_t_cos = np.sqrt(1 - theta_t_sin**2)
    t_p = 2*n1*theta_i_cos/(n2 * theta_i_cos + n1 * theta_t_cos)
    t_s = 2*n1*theta_i_cos/(n1 * theta_i_cos + n2 * theta_t_cos)

    r_p = (n2*theta_i_cos - n1*theta_t_cos)/(n2*theta_i_cos + n1*theta_t_cos)
    r_s = (n1*theta_i_cos - n2*theta_t_cos)/(n1*theta_i_cos + n2*theta_t_cos)

    

    r_p[NN_r] = 1.0
    r_s[NN_r] = 1.0
    t_p[NN_r] = 0.0
    t_s[NN_r] = 0.0
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
        #print('#$%^&&*&*())_')
        v_n = scalarproduct(-1,v_n)
        theta_i_cos = np.abs(theta_i_cos)
        theta_i = np.arccos(theta_i_cos)
    else:
        pass
    
    theta_i_sin = np.sqrt(1 - theta_i_cos**2)
    theta_t_sin = n1/n2*theta_i_sin
    NN_r = np.where(np.abs(theta_t_sin)>=1.0) # total reflection point
    theta_t_sin[NN_r] =1.0
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
    #print('check the sin(theta_i)')
    #print(s_A.reshape(11,11))
    threshold = 10**(-18)
    NN = np.where(s_A <= threshold)
    if NN[0].size != 0:
        #print('weird data!!!!!!!')
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

    #print('##############')
    poynting_t = poyntingVector(E_t,H_t)
    poynting_t_A = abs_v(poynting_t)
    poynting_r = poyntingVector(E_r,H_r)
    poynting_r_A = abs_v(poynting_r)
    #print('check energy conservation!')
    #print('check the poynting vector')  
    #print(poynting_i_A.max(),poynting_i_A.min())
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


### Fresnel coefficients for surface with AR coating
def Creat_Fresnel_coeffi_AR(theta_i,t_p,r_p,t_s,r_s,n1,n2):
    ### To be noted that t and r are the coefficients from air to silicon.
    tp_AR = CubicSpline(theta_i,t_p)
    rp_AR = CubicSpline(theta_i,r_p)
    ts_AR = CubicSpline(theta_i,t_s)
    rs_AR = CubicSpline(theta_i,r_s)
    def Fresnel_coeffi_AR1(theta):
        t_p = tp_AR(theta)
        t_s = ts_AR(theta)
        r_p = rp_AR(theta)
        r_s = rs_AR(theta)
        return t_p,t_s,r_p,r_s
    def Fresnel_coeffi_AR2(theta):
        theta_t_sin = n2*np.sin(theta)/n1
        NN_t = np.where(np.abs(theta_t_sin) >= 1.0) # total reflection point
        #print(NN_t)
        #print('**************')
        theta_t_sin[NN_t] = 1.0
        theta_t = np.arcsin(theta_t_sin)
        factor = (n2/n1)* (np.cos(theta)/np.cos(theta_t))
        factor[NN_t] = 0.0
        t_p = tp_AR(theta_t) * factor
        t_s = ts_AR(theta_t) * factor
        r_p = rp_AR(theta_t)
        r_s = rs_AR(theta_t)
        r_p[NN_t] = 1.0 * np.exp(1j*np.pi)
        r_s[NN_t] = 1.0 * np.exp(1j*np.pi)
        return t_p,t_s,r_p,r_s
    return Fresnel_coeffi_AR1,Fresnel_coeffi_AR2

def read_Fresnel_coeffi_AR(filename, groupname, n1, n2):
    with h5py.File(filename, 'r') as f:
        if groupname in f:
            group = f[groupname]
            theta_i = group['theta'][:]
            theta_t = np.arcsin(n1/n2 * np.sin(theta_i))
            tp = group['tp'][:]
            rp = group['rp'][:]
            ts = group['ts'][:]
            rs = group['rs'][:]
            factor = np.sqrt(n1 * np.cos(theta_i) / n2 /np.cos(theta_t))
            tp = tp * factor
            ts = ts * factor
            Fresnel_coeffi_AR1,Fresnel_coeffi_AR2 = Creat_Fresnel_coeffi_AR(theta_i,tp,rp,ts,rs,n1,n2)
            return Fresnel_coeffi_AR1, Fresnel_coeffi_AR2
        else:
            print(f"Group '{groupname}' not found in the file.")
            return None,None


def calculate_Field_T_R_AR(n1,n2,
                           v_n,
                           E,H,
                           AR):
    # calculate poynting vector,
    # here assuming wave vector k has same direction with poynting vector.
    poynting_i = poyntingVector(E,H)
    poynting_i_A = abs_v(poynting_i)
    k_i = scalarproduct(1/poynting_i_A,poynting_i)

    # 1. incident angle
    theta_i_cos = dotproduct(v_n,k_i)
    theta_i = np.arccos(theta_i_cos)
    if np.sum(theta_i_cos > 0) < np.sum(theta_i_cos < 0):
        v_n = scalarproduct(-1,v_n)
        theta_i_cos = np.abs(theta_i_cos)
        theta_i = np.arccos(theta_i_cos)
    else:
        pass
    
    theta_i_sin = np.sqrt(1 - theta_i_cos**2)
    theta_t_sin = n1/n2*theta_i_sin
    NN_r = np.where(np.abs(theta_t_sin)>=1.0) # total reflection point
    theta_t_sin[NN_r] =1.0
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
    #print('check the sin(theta_i)')
    #print(s_A.reshape(11,11))
    threshold = 10**(-18)
    NN = np.where(s_A <= threshold)
    if NN[0].size != 0:
        #print('weird data!!!!!!!')
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
    t_p,t_s,r_p,r_s = AR(np.arccos(np.abs(theta_i_cos)))
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
    # get H-field 
    H_r = scalarproduct(n1,crossproduct(k_r,E_r))
    H_t = scalarproduct(n2,crossproduct(k_t,E_t))

    #print('##############')
    poynting_t = poyntingVector(E_t,H_t)
    poynting_t_A = abs_v(poynting_t)
    poynting_r = poyntingVector(E_r,H_r)
    poynting_r_A = abs_v(poynting_r)
    #print('check energy conservation!')
    #print('check the poynting vector')  
    #print(poynting_i_A.max(),poynting_i_A.min())
    
    return E_t,E_r,H_t,H_r, poynting_i, n2/n1*theta_t_cos/theta_i_cos*(t_p**2+t_s**2)/2, (r_p**2+r_s**2)/2,NN