#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np;
import torch as T
import copy


# In[ ]:


'''
1. define a vector
'''
class vector():
    def __init__(self):
        self.x=np.array([])
        self.y=np.array([])
        self.z=np.array([])
    def tocoordsys(self,matrix=None):
        if matrix is None:
            self.x=self.x
            self.y=self.y
            self.z=self.z
        else:
            data=np.matmul(matrix,np.concatenate((self.x,self.y,self.z)).reshape(3,-1))
            self.x=data[0,...]
            self.y=data[1,...]
            self.z=data[2,...]
    def totensor(self,device = 'cpu'):
        self.x = T.tensor(self.x).to(device)
        self.y = T.tensor(self.y).to(device)
        self.z = T.tensor(self.z).to(device)
    def Tensor2np(self):
        if type(self.x).__module__==T.__name__:
            self.x=self.x.cpu().numpy() 
        if type(self.y).__module__==T.__name__:
            self.y=self.y.cpu().numpy()
        if type(self.z).__module__==T.__name__:
            self.z=self.z.cpu().numpy()
        else:
            pass;
        


'''
2. a field vector
'''
class Fvector():
    def __init__(self):
        self.x=np.array([],dtype=complex);
        self.y=np.array([],dtype=complex);
        self.z=np.array([],dtype=complex);

'''
3. vector operations
'''
def abs_v(A):
    Power = A.x**2 + A.y*A.y + A.z*A.z
    if isinstance(A.x,np.ndarray):
        return np.sqrt(Power)
    elif isinstance(A.x, T.Tensor):
        return T.sqrt(Power)
def abs_v_Field(A):
    Amp =np.sqrt(np.abs(A.x)**2 + np.abs(A.y)**2 + np.abs(A.z)**2)
    phase = np.angle(A.x+A.y+A.z)
    return Amp*np.exp(1j*phase)
    
def dotproduct(A,B):
    scalar = A.x.ravel()*B.x.ravel()+A.y.ravel()*B.y.ravel()+A.z.ravel()*B.z.ravel()
    return scalar

def crossproduct(A,B):
    D = vector()
    if isinstance(A.x, np.ndarray):
        A=np.append(np.append(A.x,A.y),A.z).reshape(3,-1).T
        B=np.append(np.append(B.x,B.y),B.z).reshape(3,-1).T
        C=np.cross(A,B)
        D.x=C[:,0]
        D.y=C[:,1]
        D.z=C[:,2]
    elif isinstance(A.x,T.Tensor):
        A = T.cat((A.x.ravel(),A.y.ravel(),A.z.ravel())).reshape(3,-1)
        B = T.cat((B.x.ravel(),B.y.ravel(),B.z.ravel())).reshape(3,-1)
        if (A.dtype == T.cdouble) or (B.dtype == T.cdouble):
            C = T.cross(A.type(T.cdouble),B.type(T.cdouble),dim=0)
        else:
            C = T.cross(A,B,dim=0)
        D.x=C[0,:]
        D.y=C[1,:]
        D.z=C[2,:]
    else:
        print('The input data should be np.ndarray or T.Tensor')
    return D


"""
def crossproduct(A,B):
    if type(A) != type(B):
        print('Please make sure the type of two variables are same!!!')
        return False
    else:
        D = vector()
        if isinstance(A.x, np.ndarray):
            A=np.append(np.append(A.x,A.y),A.z).reshape(3,-1).T
            B=np.append(np.append(B.x,B.y),B.z).reshape(3,-1).T
            C=np.cross(A,B,axisc=0)
            D.x=C[0,:]
            D.y=C[1,:]
            D.z=C[2,:]
        elif isinstance(A.x,T.Tensor):
            A = T.cat((A.x.ravel(),A.y.ravel(),A.z.ravel())).reshape(3,-1)
            B = T.cat((B.x.ravel(),B.y.ravel(),B.z.ravel())).reshape(3,-1)
            C = T.cross(A,B)
            D.x=C[0,:]
            D.y=C[1,:]
            D.z=C[2,:]
        else:
            print('The input data should be np.ndarray or T.Tensor')
        return D
"""
def scalarproduct(k,A):
    B=copy.copy(A)
    #B=vector()
    B.x=k*A.x.ravel()
    B.y=k*A.y.ravel()
    B.z=k*A.z.ravel()  
    return B

def sumvector(A,B):
    C=vector()
    C.x=A.x.ravel()+B.x.ravel()
    C.y=A.y.ravel()+B.y.ravel()
    C.z=A.z.ravel()+B.z.ravel()
    return C  

def CO(theta,phi):
    r0=vector();
    theta0=vector();
    PHi0=vector();
    r0.x=np.sin(theta)*np.cos(phi);
    r0.y=np.sin(theta)*np.sin(phi);
    r0.z=np.cos(theta);
    
    theta0.x=np.cos(theta)*np.cos(phi);
    theta0.y=np.cos(theta)*np.sin(phi);
    theta0.z=-np.sin(theta);
    
    PHi0.x=-np.sin(phi)
    PHi0.y=np.cos(phi);
    PHi0.z=np.zeros(phi.size);
    
    co=sumvector(scalarproduct(np.cos(phi),theta0),scalarproduct(-np.sin(phi),PHi0));
    cx=sumvector(scalarproduct(np.sin(phi),theta0),scalarproduct(np.cos(phi),PHi0));
    crho=r0;
    
    return co,cx,crho;