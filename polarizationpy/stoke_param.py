import os
import h5py as h5
import numpy as np

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