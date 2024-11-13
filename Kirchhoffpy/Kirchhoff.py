#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np;
import torch as T;
import copy;
import time;



# In[21]:


'''
1. def a class for complex values;
'''
class Complex:
    def __init__(self):
        self.real=np.array([]);
        self.imag=np.array([]);
        
    def np2Tensor(self,DEVICE):
        if type(self.real).__module__ == np.__name__:
            self.real=T.tensor(self.real).to(DEVICE).clone();
            self.imag=T.tensor(self.imag).to(DEVICE).clone();
        else:
            self.real=self.real.to(DEVICE).clone();
            self.imag=self.imag.to(DEVICE).clone();

# In[22]:


'''
2. define a Kirchhoff intergration
'''
def PO_scalar(m1,m1_n,m1_dA,m2,cos_i,Field_in,k,Keepmatrix=False):
    # output field:
    Field=Complex();
    Field.real=np.zeros(m2.x.size);
    Field.imag=np.zeros(m2.x.size);
    # the matrix
    Matrix1=Complex();
    if Keepmatrix:
        Matrix1.real=np.zeros((m2.x.size,m1.x.size));
        Matrix1.imag=np.zeros((m2.x.size,m1.x.size));
        #Matrix1.np2Tensor();
        for i in range(m2.x.size):
            x=m2.x[i]-m1.x.reshape(1,-1);
            y=m2.y[i]-m1.y.reshape(1,-1);
            z=m2.z[i]-m1.z.reshape(1,-1);
            r=np.sqrt(x**2+y**2+z**2);
            cos_r=np.abs(x*m1_n.x.reshape(1,-1)+y*m1_n.y.reshape(1,-1)+z*m1_n.z.reshape(1,-1))/r; 
            cos=(np.abs(cos_r)+np.abs(cos_i.reshape(1,-1)))/2;

            if i==int(m2.x.size/2):
                COS_R=cos_r;
            if cos_i.size==1:
                cos=1;
               
            Amp=1/r*m1_n.N*m1_dA/2/np.pi*np.abs(k)*cos;
            
            phase=-k*r;
        
            Matrix1.real[i,...]=Amp*np.cos(phase);
            Matrix1.imag[i,...]=Amp*np.sin(phase);
            Field.real[i]=(Matrix1.real[i,...]*Field_in.real.reshape(1,-1)-Matrix1.imag[i,...]*Field_in.imag.reshape(1,-1)).sum();
            Field.imag[i]=(Matrix1.real[i,...]*Field_in.imag.reshape(1,-1)+Matrix1.imag[i,...]*Field_in.real.reshape(1,-1)).sum();
        
    else:
        
        for i in range(m2.x.size):
            x=m2.x[i]-m1.x.reshape(1,-1);
            y=m2.y[i]-m1.y.reshape(1,-1);
            z=m2.z[i]-m1.z.reshape(1,-1);
            r=np.sqrt(x**2+y**2+z**2);
            cos_r=np.abs(x*m1_n.x.reshape(1,-1)+y*m1_n.y.reshape(1,-1)+z*m1_n.z.reshape(1,-1))/r;
            cos=((cos_r)+np.abs(cos_i.reshape(1,-1)))/2;
            
            if i==int(m2.x.size/2):
                COS_R=cos_r;
            if cos_i.size==1:
                cos=1;

            Amp=1/r*m1_n.N*m1_dA/2/np.pi*np.abs(k)*cos;        
            phase=-k*r;
        
            Matrix1.real=Amp*np.cos(phase);
            Matrix1.imag=Amp*np.sin(phase);
            Field.real[i]=(Matrix1.real*Field_in.real.ravel()-Matrix1.imag*Field_in.imag.ravel()).sum();
            Field.imag[i]=(Matrix1.real*Field_in.imag.ravel()+Matrix1.imag*Field_in.real.ravel()).sum();
            
    return Matrix1,Field,COS_R;
        
    
        
    
    

