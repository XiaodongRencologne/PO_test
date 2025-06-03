import numpy as np

def Fresnel_coeffi_0(n1,n2,angle_i):
    sin_angle_t = n1/n2*np.sin(angle_i)
    cos_angle_t = np.sqrt(1-sin_angle_t**2)

    R_p = (n2*np.cos(angle_i) - n1*cos_angle_t)/\
          (n2*np.cos(angle_i) + n1*cos_angle_t)
    R_s = (n1*np.cos(angle_i) - n2*cos_angle_t)/\
          (n1*np.cos(angle_i) + n2*cos_angle_t)

    T_p = 2*n1*np.cos(angle_i)/(n2*np.cos(angle_i) + n1*cos_angle_t)
    T_s = 2*n1*np.cos(angle_i)/(n1*np.cos(angle_i) + n2*cos_angle_t)

    return R_p, R_s, T_p, T_s

def film(Thickness,
         n1,n2,n3,
         angle_i,
         Lambda,
         polarization = 'perpendicular'):
    angle2 = np.arcsin(n1/n2*np.sin(angle_i))
    angle3 = np.arcsin(n2/n3*n1/n2*np.sin(angle_i))
    beta = 2* np.pi / Lambda * n2 * Thickness * np.cos(angle2)
    p1 = n1 * np.cos(angle_i) 
    p2 = n2 * np.cos(angle2)
    p3 = n3 * np.cos(angle3)
    rp1, rs1, tp1, ts1 = Fresnel_coeffi_0(n1,n2,angle_i)
    R12 = {'s': rs1,
           'p': rp1}
    T12 = {'s': ts1,
           'p': tp1}
    rp2, rs2, tp2, ts2 = Fresnel_coeffi_0(n2,n3,angle2)

    R23 = {'s': rs2,
           'p': rp2}
    T23 = {'s': ts2,
           'p': tp2}
    
    if polarization == 'perpendicular':
        polar = 's'
    else:
        polar = 'p'

    r12 = R12[polar]
    r23 = R23[polar]
    t12 = T12[polar]
    t23 = T23[polar]
    
    r = (r12 + r23*np.exp(2j*beta))/(1 + r12*r23*np.exp(2j*beta))
    t = t12*t23*np.exp(1j*beta)/(1+r12*r23*np.exp(2j*beta))

    R = np.abs(r)**2
    T = p3/p1 *np.abs(t)**2
    return R, T

def Fresnel_coeffi(n1,n2,angle_i):
    theta_i_cos = np.cos(angle_i)
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


def Fresnel_coeffi_power(n1,n2,angle_i):
    theta_i_cos = np.cos(angle_i)
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
    r_p = r_p**2
    r_s = r_s**2
    factor = n2 / n1 * theta_t_cos / theta_i_cos
    t_p = t_p**2 * factor
    t_s = t_s**2 * factor
    return t_p,t_s,r_p,r_s

def critical_angle(n1, n2):
    """
    Calculate the critical angle for total internal reflection.
    
    Parameters:
    n1 (float): Refractive index of the first medium.
    n2 (float): Refractive index of the second medium.
    
    Returns:
    float: Critical angle in radians.
    """
    if n1 <= n2:
        return None  # Total internal reflection cannot occur
    return np.arcsin(n2 / n1)  # Critical angle in radians

def brewster_angle(n1, n2):
    """
    Calculate the Brewster angle.
    
    Parameters:
    n1 (float): Refractive index of the first medium.
    n2 (float): Refractive index of the second medium.
    
    Returns:
    float: Brewster angle in radians.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Refractive indices must be positive.")
    return np.arctan(n2 / n1)  # Brewster angle in radians