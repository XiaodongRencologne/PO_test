#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np;


# In[ ]:


'''
1. define a vector
'''
class vector():
    def __init__(self):
        self.x=np.array([]);
        self.y=np.array([]);
        self.z=np.array([]);
    def tocoordsys(self,matrix=None):
        if matrix==None:
            self.gx=self.x;
            self.gy=self.y;
            self.gz=self.z;
        else:
            data=np.matmul(matrix,np.concatenate((self.x,self.y,self.z)).reshape(3,-1));
            self.gx=data[0,...]
            self.gy=data[1,...]
            self.gz=data[2,...]


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
def dotproduct(A,B):
    return A.x.ravel()*B.x.ravel()+A.y.ravel()*B.y.ravel()+A.z.ravel()*B.z.ravel();

def crossproduct(A,B):
    A=np.append(np.append(A.x,A.y),A.z).reshape(3,-1).T;
    B=np.append(np.append(B.x,B.y),B.z).reshape(3,-1).T;
    C=np.cross(A,B);
    D=vector();
    D.x=C[...,0]
    D.y=C[...,1];
    D.z=C[...,2];
    return D;

def scalarproduct(k,A):
    B=vector();
    B.x=k*A.x;
    B.y=k*A.y;
    B.z=k*A.z;    
    return B;

def sumvector(A,B):
    C=vector();
    C.x=A.x+B.x;
    C.y=A.y+B.y;
    C.z=A.z+B.z;
    return C;    

