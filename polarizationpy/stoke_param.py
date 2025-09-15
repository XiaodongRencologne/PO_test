import os
import h5py as h5
import numpy as np

def Jone_Matrix(E_co_0,E_cx_0,
                E_co_90,E_cx_90,):
    Point_N = E_co_0.size
    J_uv = np.concatenate((E_co_0.ravel(),E_cx_0.ravel(),
                           E_co_90.ravel(),E_cx_90.ravel())).reshape(2,2,Point_N)
    J_uv = np.moveaxis(J_uv,-1,0)
    return J_uv

def Mueller_Matrix(J_uv):
    A = np.array([[1,0,0,1],
                  [1,0,0,-1],
                  [0,1,1,0],
                  [0,1j, -1j,0]],dtype=complex)
    A_inv = np.array([[1,1,0,0],
                      [0,0,1,-1j],
                      [0,0,1,1j],
                      [1,-1,0,0]],dtype = complex)*1/2
    K = np.array([np.kron(J_uv[i], np.conjugate(J_uv[i])) for i in range(J_uv.shape[0])])
    M = A @ K @ A_inv
    #M = np.einsum('ij,njk,kl->nil', A, K, A_inv, optimize=True)
    return M.real

def unpolarized_s_beam(M,I_sky):
    I_sky = I_sky.ravel()
    I_beam = M[:,0,0].ravel() * I_sky
    Q_beam = M[:,1,0].ravel() * I_sky
    U_beam = M[:,2,0].ravel() * I_sky
    V_beam = M[:,3,0].ravel() * I_sky

    Imax = I_beam.max()
    I_beam, Q_beam, U_beam, V_beam = I_beam/Imax, Q_beam/Imax, U_beam/Imax, V_beam/Imax
    I = I_beam.sum()
    Q = Q_beam.sum()/I
    U = U_beam.sum()/I
    V = V_beam.sum()/I
    print('I max:',Imax)
    print('I sum:',I)
    print('IQUV:',1,Q,U,V)
    print(np.sqrt(Q**2+U**2))
    print('T->P leakage:',10*np.log10(np.sqrt(Q**2+U**2)),'dB')
    return I_beam, Q_beam, U_beam, V_beam



def stokes_beams(E_co,E_cx):
    CO = np.abs(E_co)**2
    CX = np.abs(E_cx)**2
    I = CO +CX
    Q = CO - CX
    UV = np.conjugate(E_co)* E_cx
    U = 2 * UV.real
    V = 2 * UV.imag
    Max = I.max()
    I, Q, U,V = I/Max, Q/Max, U/Max, V/Max
    print(Q.sum(), U.sum(),V.sum())
    return I,Q,U,V