import os
import h5py as h5
import numpy as np
from scipy.optimize import minimize


def polarization_angle(beam):

    def rotation_angle0(phi):
        Rotation_Mat = np.array([[np.cos(phi[0]),np.sin(phi[0])],
                                 [-np.sin(phi[0]), np.cos(phi[0])]])
        beam_out = np.matmul(Rotation_Mat, beam)
        #beam_out = np.abs(beam_out)**2
        return beam_out
    
    def lossfuc(phi):
        beam_out = rotation_angle0(phi)
        beam_out = np.abs(beam_out)**2
        #Q = 1/ np.abs((beam_out[0,:].sum() - beam_out[1,:].sum()))
        Q = beam_out[1,:].sum()/beam_out[0,:].sum()
        return Q
    results = minimize(lossfuc,0 ,method = 'BFGS' , jac = False, tol = 10**(-8))
    print(results.success, results.fun)
    return results

def rotation_angle(phi,beam0):
    Rotation_Mat = np.array([[np.cos(phi[0]),np.sin(phi[0])],
                                 [-np.sin(phi[0]), np.cos(phi[0])]])
    beam_out = np.matmul(Rotation_Mat, beam0)
    #beam_out = np.abs(beam_out)**2
    return beam_out


def polarization_angle_method1(beam):

    def rotation_angle0(phi):
        Rotation_Mat = np.array([[np.cos(phi[0]),np.sin(phi[0])],
                                 [-np.sin(phi[0]), np.cos(phi[0])]])
        beam_out = np.matmul(Rotation_Mat, beam)
        #beam_out = np.abs(beam_out)**2
        return beam_out
    
    def lossfuc(phi):
        beam_out = rotation_angle0(phi)
        beam_out = np.abs(beam_out)**2
        #Q = 1/ np.abs((beam_out[0,:].sum() - beam_out[1,:].sum()))
        Max = beam_out[0,:].max()
        N_max = np.where(beam_out[0,:] == Max)
        Q = beam_out[1,:][N_max]/Max
        return Q
    results = minimize(lossfuc,0 ,method = 'BFGS' , jac = False, tol = 10**(-8))
    print(results.success, results.fun)
    return results


def polarization_angle_method2(beam):

    def rotation_angle0(phi):
        Rotation_Mat = np.array([[np.cos(phi[0]),np.sin(phi[0])],
                                 [-np.sin(phi[0]), np.cos(phi[0])]])
        beam_out = np.matmul(Rotation_Mat, beam)
        #beam_out = np.abs(beam_out)**2
        return beam_out
    
    def lossfuc(phi):
        beam_out = rotation_angle0(phi)
        beam_out = np.abs(beam_out)**2
        #Q = 1/ np.abs((beam_out[0,:].sum() - beam_out[1,:].sum()))
        Q = beam_out[1,:].max()/beam_out[0,:].max()
        return Q
    results = minimize(lossfuc,0 ,method = 'BFGS' , jac = False, tol = 10**(-8))
    print(results.success, results.fun)
    return results