import numpy as np

def Fresnel_coeffi(n1,n2,angle_i):
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
    rp1, rs1, tp1, ts1 = Fresnel_coeffi(n1,n2,angle_i)
    R12 = {'s': rs1,
           'p': rp1}
    T12 = {'s': ts1,
           'p': tp1}
    rp2, rs2, tp2, ts2 = Fresnel_coeffi(n2,n3,angle2)

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