import numpy as np

#### coordinates points transform.
def euler2mat(ai,al,ak,axes='xyz'):
    '''
    Calculate coordinate transform matrix based on eular angles
    '''
    angle=[ai,al,ak]

    axis={'x': lambda phi: np.array([[1.0,0.0,0.0],[0.0,np.cos(phi),np.sin(phi)],[0.0,-np.sin(phi),np.cos(phi)]]).ravel().reshape(3,3),
          'y': lambda phi: np.array([[np.cos(phi),0.0,-np.sin(phi)],[0.0,1.0,0.0],[np.sin(phi),0.0,np.cos(phi)]]).ravel().reshape(3,3),
          'z': lambda phi: np.array([[np.cos(phi),np.sin(phi),0.0],[-np.sin(phi),np.cos(phi),0.0],[0.0,0.0,1.0]]).ravel().reshape(3,3)
      }
    M=np.eye(3)
    i=0
    for n in axes:
        M=np.matmul(axis[n](angle[i]),M) 
        i+=1
    return M

def cartesian2spherical(x,y,z):
    '''
    Convert cartesian coordinates into spherical coordinates
    '''
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arccos(z/r)
    phi=np.arctan2(y,x)
    return r, theta, phi

def cartesian2cylinder(x,y,z):
    '''
    convert cartesian coordinates to cylinder coordinates.
    '''
    pho=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    z=z
    
    return pho, phi, z
    

def Transform_local2global (angle,origin,local,axes='xyz'):
    '''
    convert corrdinates from local coordinate system to global system
    '''
    origin=np.array(origin)
    L=np.append([local.x,local.y],[local.z],axis=0)
    mat=euler2mat(angle[0],angle[1],angle[2],axes=axes) 
    mat=np.transpose(mat)
    G=np.matmul(mat,L) 
    G=G+origin.reshape(-1,1)
    return G[0,:], G[1,:], G[2,:]

def Transform_global2local (angle,origin,G,axes='xyz'):  
    '''
    convert corrdinates from local coordinate system to global system
    '''
    origin=np.array(origin)
    g=np.append([G.x,G.y],[G.z],axis=0)
    g=g-origin.reshape(-1,1)
    mat=euler2mat(angle[0],angle[1],angle[2],axes=axes)
    local=np.matmul(mat,g)    

    return local[0,...], local[1,...], local[2,...]
