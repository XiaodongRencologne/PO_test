import numpy as np
from .coordinate_operations import Coord
from .coordinate import coord_sys
from .Vopy import vector

class Spherical_grd():
    def __init__(self,
                 coord_sys,
                 x0,
                 y0,
                 x_size,
                 y_size,
                 Nx,Ny,
                 Type = 'ELoverAz', 
                 far_near = 'far',
                 distance = 0,
                 ):
        Type = Type.lower()
        self.far_near = far_near
        self.coord_sys = coord_sys
        self.E=vector()
        self.F=vector()
        self.grid = Coord()
        self.x = np.linspace(x0-x_size/2,x0+x_size/2,Nx)
        self.y = np.linspace(y0-y_size/2,y0+y_size/2,Ny)
        X,Y = np.meshgrid(self.x,self.y)
        X = X.ravel()
        Y = Y.ravel()
        Grid_type={'uv':      lambda x,y: (x,y,np.sqrt(1-(x**2+y**2))),
                   'eloveraz':lambda x,y: (-np.sin(x/180*np.pi)*np.cos(y/180*np.pi),
                                           np.sin(y/180*np.pi),
                                           np.cos(x/180*np.pi)*np.cos(y/180*np.pi))
              }
        self.grid.x, self.grid.y, self.grid.z = Grid_type[Type](X,Y)

        
        if far_near == 'far':
            pass
        elif far_near =='near':
            self.grid.x = distance * self.grid.x
            self.grid.y = distance * self.grid.y
            self.grid.z =  distance * self.grid.z
            pass
        else:
            print('Error input!')