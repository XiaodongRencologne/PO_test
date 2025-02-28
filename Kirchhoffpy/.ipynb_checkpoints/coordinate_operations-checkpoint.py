#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''

The package provides a few functions used to realsize coordinates translations
'''
import numpy as np;
import torch as T;
import transforms3d;
import copy


# In[2]:

class Coord:
    def __init__(self):
        # local coordinates
        self.x=np.array([]);
        self.y=np.array([]);
        self.z=np.array([]);
        self.N=np.array([]);

    def np2Tensor(self,DEVICE = 'cpu'):
        '''DEVICE=T.device('cpu') or T.device('cude:0')'''
        if type(self.x).__module__ == np.__name__:
            self.x=T.tensor(self.x,dtype=T.float64).to(DEVICE).clone()
        elif type(self.x).__module__==T.__name__:
            self.x=self.x.to(DEVICE);            
        if type(self.y).__module__ == np.__name__:
            self.y=T.tensor(self.y,dtype=T.float64).to(DEVICE).clone()
        elif type(self.y).__module__==T.__name__:
            self.y=self.y.to(DEVICE)
        if type(self.z).__module__ == np.__name__:
            self.z=T.tensor(self.z,dtype=T.float64).to(DEVICE).clone()
        elif type(self.z).__module__==T.__name__:
            self.z=self.z.to(DEVICE)
        if type(self.N).__module__ == np.__name__:
            self.N=T.tensor(self.N,dtype=T.float64).to(DEVICE).clone()
        elif type(self.N).__module__==T.__name__:
            self.N=self.N.to(DEVICE)
        else:
            print('The input data is wrong')

    def Tensor2np(self):
        if type(self.x).__module__==T.__name__:
            self.x=self.x.cpu().numpy();
        if type(self.y).__module__==T.__name__:
            self.y=self.y.cpu().numpy();
        if type(self.z).__module__==T.__name__:
            self.z=self.z.cpu().numpy();
        if type(self.N).__module__==T.__name__:
            self.N=self.N.cpu().numpy();
        else:
            pass;


    

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
    #g=Coord();
    g = copy.copy(local)
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
    #l=Coord();
    l = copy.copy(G)
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
    

