# %%
import numpy as np;
import copy
from transform import euler2mat, cartesian2spherical, cartesian2cylinder

# %%
class _global_coord_sys():
    def __init__(self):
        self.origin=np.zeros((3,1))
        self.origin_g=np.zeros((3,1))
        self.mat_l_g=np.eye(3)
        self.mat_r_l=np.eye(3)
        self.mat_l_r=np.eye(3)
        self.name = 'mechanical_axis.cs'
    def _toRef_coord(self,x,y,z):
        return x,y,z
    def _toLocal_coord(self,x,y,z):
        return x,y,z
    def _toGlobal_coord(self,x,y,z):
        return x,y,z
    def Global_to_local(self,x,y,z):
        return x,y,z
    def _toSpherical(self,x,y,z):
        return cartesian2spherical(x,y,z)  
    def _toCylinder(self,x,y,z):
        return cartesian2cylinder(x,y,z)
    
global_coord=_global_coord_sys()


# %%
class coord_sys():
    '''
    define a coordinate system by giving a reference coord 'ref_coord', origin 'origin' in 
    the reference coord, roation angles 'angle=[0,0,0]' and rotating axes 'xyz' based on euler angles.
    The default reference coord is pre-defined global coordinate system.
    '''
    def __init__(self,origin,angle,axes='xyz',ref_coord=global_coord,name='coor_sys'):
        self.origin=np.array(origin).reshape(3,1)

        self.mat_r_l=euler2mat(angle[0],angle[1],angle[2],axes=axes)
        self.mat_l_r=np.transpose(self.mat_r_l)
        self.mat_l_g=np.matmul(ref_coord.mat_l_g,self.mat_l_r)
        self.mat_g_l=np.transpose(self.mat_l_g)

        # origin in global coordinate system.
        self.origin_g=ref_coord.origin_g+np.matmul(ref_coord.mat_l_g,self.origin)

        self.name = name
        self.ref_name = ref_coord.name

    def _toRef_coord(self,x,y,z):
        '''
        convert coordinates from local to reference system
        '''
        xyz=np.append([x,y],[z],axis=0)
        xyz=np.matmul(self.mat_l_r,xyz)+self.origin
        return xyz[0,:], xyz[1,:], xyz[2,:]
    
    def _toLocal_coord(self,x,y,z):
        '''
        coordinates from reference coord to local coord, 
        commonly this is function is useless.
        '''
        xyz=np.append([x,y],[z],axis=0)
        xyz=xyz-self.origin
        xyz=np.matmul(self.mat_r_l,xyz)
        return xyz[0,:], xyz[1,:], xyz[2,:]
    
    def _toGlobal_coord(self,x,y,z):
        '''
        from local to global.
        '''
        xyz=np.append([x,y],[z],axis=0)
        xyz=np.matmul(self.mat_l_g,xyz)+self.origin_g
        return xyz[0,:], xyz[1,:], xyz[2,:]

    def Global_to_local(self,x,y,z):
        xyz=np.append([x,y],[z],axis=0)
        xyz=xyz-self.origin_g
        xyz=np.matmul(self.mat_g_l,xyz)
        return xyz[0,:], xyz[1,:], xyz[2,:]

    def _toSpherical(self,x,y,z):
        r, theta, phi = cartesian2spherical(x,y,z)
        return r, theta, phi
    
    def _toCylinder(self,x,y,z):
        pho, phi, z = cartesian2cylinder(x,y,z)
        return pho, phi, z
