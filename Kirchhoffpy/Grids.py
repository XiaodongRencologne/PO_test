import numpy as np
from .coordinate_operations import Coord

def plane(sizex ,sizey, Nx, Ny):
    plane1 = Coord()
    x = np.linspace(-sizex,sizex,Nx)
    y = np.linspace(-sizey,sizey, Ny)
    plane1.x, plane1.y = np.meshgrid(x,y)
    plane1.x = plane1.x.ravel()
    plane1.y = plane1.y.ravel()
    plane1.z = np.zeros(plane1.x.shape)
    P_n = Coord()
    P_n.x = np.zeros(plane1.x.shape)
    P_n.y = np.zeros(plane1.x.shape)
    P_n.z = np.ones(plane1.x.shape)
    
    return plane1, P_n