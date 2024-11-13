#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''

The package provides a few functions used to realsize coordinates translations
'''
import numpy as np;
import torch as T;
import transforms3d;


# In[2]:

class Coord:
    def __init__(self):
        # local coordinates
        self.x=np.array([]);
        self.y=np.array([]);
        self.z=np.array([]);
        self.N=np.array([]);
        

    def np2Tensor(self,DEVICE):
        # numpy to tensor;
        self.x=T.tensor(self.x).to(DEVICE);
        self.y=T.tensor(self.y).to(DEVICE);
        self.z=T.tensor(self.z).to(DEVICE);


    

'''
coordinates transformation, from local coordinates to global coordinates;
'''        
def Transform_local2global (angle,displacement,local):
    displacement=np.array(displacement);
    L=np.append([local.x,local.y],[local.z],axis=0)
    mat=transforms3d.euler.euler2mat(-angle[0],-angle[1],-angle[2]);  
    mat=np.transpose(mat);
    G=np.matmul(mat,L);   
    G=G+displacement.reshape(-1,1);
    g=Coord();
    g.x=G[0,...];
    g.y=G[1,...];
    g.z=G[2,...];
    g.N=local.N;
    return g;

def Transform_global2local (angle,displacement,G):  
    displacement=np.array(displacement);
    g=np.append([G.x,G.y],[G.z],axis=0)
    g=g-displacement.reshape(-1,1);
    mat=transforms3d.euler.euler2mat(-angle[0],-angle[1],-angle[2]);
    
    local=np.matmul(mat,g);      
    l=Coord();
    l.x=local[0,...];
    l.y=local[1,...];
    l.z=local[2,...];
    l.N=G.N;
    return l;
'''
get the spherical coordinates from cartesian coordinates;
'''
def cartesian_to_spherical(x,y,z):
    
    r=np.sqrt(x**2+y**2+z**2);
    theta=np.arccos(z/r);
    phi=np.arctan2(y,x);
    
    return r,theta,phi;
    

