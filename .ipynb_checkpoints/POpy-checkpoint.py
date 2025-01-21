#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
import torch as T;
from numba import njit, prange

import copy;
import time;


# In[28]:


'''
1. Define Electromagnetic-Field Data, which is a vector and each component is a complex value
'''
class Complex():
    '''
    field is combination of real and imag parts to show the phase informations
    '''
    
    def __init__(self):
        self.real=np.array([]);
        self.imag=np.array([]);
        
    def np2Tensor(self,DEVICE=T.device('cpu')):
        '''DEVICE=T.device('cpu') or T.device('cude:0')'''
        if type(self.real).__module__ == np.__name__:
            self.real=T.tensor(self.real).to(DEVICE).clone();
        elif type(self.real).__module__==T.__name__:
            self.real=self.real.to(DEVICE);            
        if type(self.imag).__module__ == np.__name__:
            self.imag=T.tensor(self.imag).to(DEVICE).clone();
        elif type(self.imag).__module__==T.__name__:
            self.imag=self.imag.to(DEVICE);
        else:
            print('The input data is wrong')
            
    def Tensor2np(self):
        if type(self.real).__module__==T.__name__:
            self.real=self.real.cpu().numpy();
            
        if type(self.imag).__module__==T.__name__:
            self.imag=self.imag.cpu().numpy();
        else:
            pass;

class Field_Vector():
    '''
    Field Vector Fx Fy Fz, each part is a complex value.
    '''
    def __init__(self):
        self.x=Complex();
        self.y=Complex();
        self.z=Complex();
    def np2Tensor(self,DEVICE=T.device('cpu')):
        self.x.np2Tensor(DEVICE);
        self.y.np2Tensor(DEVICE);
        self.z.np2Tensor(DEVICE);
    def Tensor2np(self):
        self.x.Tensor2np(DEVICE);
        self.y.Tensor2np(DEVICE);
        self.z.Tensor2np(DEVICE);
        

'''
2. Fresnel-Kirchhoff intergration
   2.1 'Kirchhoff' to calculate near field
   2.2 'Kirchhoff_far' used to calculate far field
'''
def Kirchhoff(face1,face1_n,face1_dS,face2,cos_in,Field1,k,Keepmatrix=False,parallel=True):
    # output field:
    Field_face2=Complex();
    Matrix=Complex();
    COS_R=1;
    
    
    ''' calculate the field including the large matrix'''
    def calculus1(x1,y1,z1,x2,y2,z2,nx1,ny1,nz1,N,ds,cos_i,Field_in_real,Field_in_imag):
        M_real=np.zeros((x2.size,x1.size));
        M_imag=np.zeros((x2.size,x1.size));
        for i in range(x2.size):
            x=x2[i]-x1.reshape(1,-1);
            y=y2[i]-y1.reshape(1,-1);
            z=z2[i]-z1.reshape(1,-1);
            r=np.sqrt(x**2+y**2+z**2);
            cos_r=np.abs(x*nx1.reshape(1,-1)+y*ny1.reshape(1,-1)+z*nz1.reshape(1,-1))/r; 
            cos=(np.abs(cos_r)+np.abs(cos_i.reshape(1,-1)))/2;
            if i==int(x2.size/2):
                COS_r=cos_r;
            if cos_i.size==1:
                cos=1;
            Amp=1/r*N*ds/2/np.pi*np.abs(k)*cos;            
            phase=-k*r;
        
            M_real[i,...]=Amp*np.cos(phase);
            M_imag[i,...]=Amp*np.sin(phase);
            Field_real[i]=(M_real[i,...]*Field_in_real.reshape(1,-1)-M_imag[i,...]*Field_in_imag.reshape(1,-1)).sum();
            Field_imag[i]=(M_real[i,...]*Field_in_imag.reshape(1,-1)+M_imag[i,...]*Field_in_real.reshape(1,-1)).sum();
        return M_real,M_imag,Field_real,Field_imag,COS_r

    '''without calculating large matrix to save memory'''
    @njit(parallel=parallel)
    def calculus2(x1,y1,z1,x2,y2,z2,nx1,ny1,nz1,N,ds,cos_i,Field_in_real,Field_in_imag):
        Field_real=np.zeros(x2.size);
        Field_imag=np.zeros(x2.size);        
        for i in prange(x2.size):
            x=x2[i]-x1.reshape(1,-1);
            y=y2[i]-y1.reshape(1,-1);
            z=z2[i]-z1.reshape(1,-1);
            r=np.sqrt(x**2+y**2+z**2);
            cos_r=np.abs(x*nx1.reshape(1,-1)+y*ny1.reshape(1,-1)+z*nz1.reshape(1,-1))/r;
            cos_r=(np.abs(cos_r)+np.abs(cos_i.reshape(1,-1)))/2                
            Amp=1/r*N*ds/2/np.pi*np.abs(k)*cos_r;            
            phase=-k*r;
            M_real=Amp*np.cos(phase);
            M_imag=Amp*np.sin(phase);
            Field_real[i]=(M_real*Field_in_real.ravel()-M_imag*Field_in_imag.ravel()).sum();
            Field_imag[i]=(M_real*Field_in_imag.ravel()+M_imag*Field_in_real.ravel()).sum();
    
        return Field_real,Field_imag;
    '''without calculating large matrix to save memory'''
    @njit(parallel=parallel)
    def calculus3(x1,y1,z1,x2,y2,z2,nx1,ny1,nz1,N,ds,cos_i,Field_in_real,Field_in_imag):
        Field_real=np.zeros(x2.size);
        Field_imag=np.zeros(x2.size);        
        for i in prange(x2.size):
            x=x2[i]-x1.reshape(1,-1);
            y=y2[i]-y1.reshape(1,-1);
            z=z2[i]-z1.reshape(1,-1);
            r=np.sqrt(x**2+y**2+z**2);               
            Amp=1/r*N*ds/2/np.pi*np.abs(k);            
            phase=-k*r;
            M_real=Amp*np.cos(phase);
            M_imag=Amp*np.sin(phase);
            Field_real[i]=(M_real*Field_in_real.ravel()-M_imag*Field_in_imag.ravel()).sum();
            Field_imag[i]=(M_real*Field_in_imag.ravel()+M_imag*Field_in_real.ravel()).sum();
    
        return Field_real,Field_imag;
    
    if Keepmatrix:
        Matrix.real,Matrix.imag,Field_face2.real,Field_face1.imag,COS_R=calculus1(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                                                  face1_n.x,face1_n.y,face1_n.z,face1_n.N,face1_dS,cos_in,Field1.real,Field1.imag);
        return Matrix,Field_face2,COS_R;
    else:
        if cos_in.size==1:
            Field_face2.real,Field_face2.imag=calculus3(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                    face1_n.x,face1_n.y,face1_n.z,face1_n.N,face1_dS,cos_in,Field1.real,Field1.imag);
        else:
            Field_face2.real,Field_face2.imag=calculus2(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                    face1_n.x,face1_n.y,face1_n.z,face1_n.N,face1_dS,cos_in,Field1.real,Field1.imag);
        return Matrix,Field_face2,COS_R;

    
'''2.2 calculate the far-field beam'''    
def Kirchhoff_far(face1,face1_n,face1_dS,face2,cos_in,Field1,k,Keepmatrix=False,parallel=True):
    # output field:
    Field_face2=Complex();
    Matrix=Complex();
    COS_R=1;    
    ''' calculate the field including the large matrix'''
    def calculus1(x1,y1,z1,x2,y2,z2,nx1,ny1,nz1,N,ds,cos_i,Field_in_real,Field_in_imag):
        M_real=np.zeros((x2.size,x1.size));
        M_imag=np.zeros((x2.size,x1.size));
        for i in range(x2.size):
            phase=k*(x2[i]*x1.reshape(1,-1)+y2[i]*y1.reshape(1,-1)+z2[i]*z1.reshape(1,-1))
            cos_r=x2[i]*nx1.reshape(1,-1)+y2[i]*ny1.reshape(1,-1)+z2[i]*nz1.reshape(1,-1)
            cos=(np.abs(cos_r)+np.abs(cos_i).reshape(1,-1))/2;           
            if i==int(x2.size/2):
                COS_r=cos_r;
            if cos_i.size==1:
                cos=1;     
            Amp=k*N*ds/2/np.pi*np.abs(k)*cos;          
            M_real[i,...]=Amp*np.cos(phase);
            M_imag[i,...]=Amp*np.sin(phase);
            Field_real[i]=(M_real[i,...]*Field_in_real.reshape(1,-1)-M_imag[i,...]*Field_in_imag.reshape(1,-1)).sum();
            Field_imag[i]=(M_real[i,...]*Field_in_imag.reshape(1,-1)+M_imag[i,...]*Field_in_real.reshape(1,-1)).sum();
        return M_real,M_imag,Field_real,Field_imag,COS_r

    '''without calculating large matrix to save memory'''
    @njit(parallel=parallel)
    def calculus2(x1,y1,z1,x2,y2,z2,nx1,ny1,nz1,N,ds,cos_i,Field_in_real,Field_in_imag):
        Field_real=np.zeros(x2.size);
        Field_imag=np.zeros(x2.size);        
        for i in prange(x2.size):
            phase=k*(x2[i]*x1.reshape(1,-1)+y2[i]*y1.reshape(1,-1)+z2[i]*z1.reshape(1,-1))
            cos_r=x2[i]*nx1.reshape(1,-1)+y2[i]*ny1.reshape(1,-1)+z2[i]*nz1.reshape(1,-1)
            cos=(np.abs(cos_r)+np.abs(cos_i).reshape(1,-1))/2;
            Amp=k*N*ds/2/np.pi*np.abs(k)*cos;                        
            M_real=Amp*np.cos(phase);
            M_imag=Amp*np.sin(phase);
            Field_real[i]=(M_real*Field_in_real.ravel()-M_imag*Field_in_imag.ravel()).sum();
            Field_imag[i]=(M_real*Field_in_imag.ravel()+M_imag*Field_in_real.ravel()).sum();
    
        return Field_real,Field_imag;
    '''without calculating large matrix to save memory'''
    @njit(parallel=parallel)
    def calculus3(x1,y1,z1,x2,y2,z2,nx1,ny1,nz1,N,ds,cos_i,Field_in_real,Field_in_imag):
        Field_real=np.zeros(x2.size);
        Field_imag=np.zeros(x2.size);        
        for i in prange(x2.size):
            phase=k*(x2[i]*x1.reshape(1,-1)+y2[i]*y1.reshape(1,-1)+z2[i]*z1.reshape(1,-1));            
            Amp=k*N*ds/2/np.pi*np.abs(k);            
            M_real=Amp*np.cos(phase);
            M_imag=Amp*np.sin(phase);
            Field_real[i]=(M_real*Field_in_real.ravel()-M_imag*Field_in_imag.ravel()).sum();
            Field_imag[i]=(M_real*Field_in_imag.ravel()+M_imag*Field_in_real.ravel()).sum();
    
        return Field_real,Field_imag;
    
    if Keepmatrix:
        Matrix.real,Matrix.imag,Field_face2.real,Field_face1.imag,COS_R=calculus1(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                                                  face1_n.x,face1_n.y,face1_n.z,face1_n.N,face1_dS,cos_in,Field1.real,Field1.imag);
        return Matrix,Field_face2,COS_R;
    else:
        if cos_in.size==1:
            Field_face2.real,Field_face2.imag=calculus3(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                    face1_n.x,face1_n.y,face1_n.z,face1_n.N,face1_dS,cos_in,Field1.real,Field1.imag);
        else:
            Field_face2.real,Field_face2.imag=calculus2(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                    face1_n.x,face1_n.y,face1_n.z,face1_n.N,face1_dS,cos_in,Field1.real,Field1.imag);
        return Matrix,Field_face2,COS_R;
