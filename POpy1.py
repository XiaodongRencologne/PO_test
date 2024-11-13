#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
import torch as T;
from numba import njit, prange;
from Vopy import vector,crossproduct,scalarproduct;

import copy;
import time;

mu=4*np.pi*10**(-7);
epsilon=8.854187817*10**(-12);
Z0=np.sqrt(mu/epsilon);


# In[ ]:


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


# In[ ]:


'''
3. Physical optics intergration
   3.1 'Physical optics' to calculate near field
   3.2 'far' used to calculate far field
'''
def PO(face1,face1_n,face1_dS,face2,Field_in_E,Field_in_H,k,parallel=True):
    # output field:
    Field_E=vector();
    Field_H=vector();    
    Je_in=scalarproduct(1,crossproduct(face1_n,Field_in_H));
    if Field_in_E==0:
        Jm_in=0;
    else:
        Jm_in=scalarproduct(1,crossproduct(face1_n,Field_in_E));
    JE=np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,-1);
    
    '''magnetic field is zero'''
    @njit(parallel=parallel)
    def calculus1(x1,y1,z1,x2,y2,z2,N,ds,Je): 
        Field_E_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_E_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_E_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        #R=np.zeros((3,x1.size));
        #he2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
        for i in prange(x2.size): 
            R=np.zeros((3,x1.size));
            R[0,...]=x2[i]-x1.ravel();
            R[1,...]=y2[i]-y1.ravel();
            R[2,...]=z2[i]-z1.ravel();
            r=np.sqrt(np.sum(R**2,axis=0));
            
            '''calculate the vector potential Ae based on induced current'''
            phase=-k*r;
            r2=(k**2)*(r**2);
            r3=(k**3)*(r**3);
            '''1'''
            ee=np.exp(1j*phase)*k**2*(Je*(1j/phase-1/r2+1j/r3)+np.sum(Je*R/r,axis=0)*R/r*(-1j/phase+3/r2-3j/r3));
            Ee=np.sum(ee*N*ds,axis=1);
            '''2'''
            he=np.exp(1j*phase)*k**2
            he1=(R/r*1/(r2)*(1-1j*phase));
            he2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
            he2[0,...]=Je[1,...]*he1[2,...]-Je[2,...]*he1[1,...];
            he2[1,...]=Je[2,...]*he1[0,...]-Je[0,...]*he1[2,...];
            he2[2,...]=Je[0,...]*he1[1,...]-Je[1,...]*he1[0,...];
            He=np.sum(he*he2*N*ds,axis=1);
            
            Field_E_x[i]=Z0/(4*np.pi)*Ee[0]
            Field_E_y[i]=Z0/(4*np.pi)*Ee[1]
            Field_E_z[i]=Z0/(4*np.pi)*Ee[2]
        
            Field_H_x[i]=1/4/np.pi*He[0]
            Field_H_y[i]=1/4/np.pi*He[1]
            Field_H_z[i]=1/4/np.pi*He[2]


        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z;
    '''Jm!=0'''
    @njit(parallel=parallel)
    def calculus2(x1,y1,z1,x2,y2,z2,N,ds,Je,Jm):
        Field_E_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_E_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_E_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        Field_H_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape);
        
        #em2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
        #he2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
        for i in prange(x2.size):
            R=np.zeros((3,x1.size));
            R[0,...]=x2[i]-x1.reshape(1,-1);
            R[1,...]=y2[i]-y1.reshape(1,-1);
            R[2,...]=z2[i]-z1.reshape(1,-1);
            
            r=np.sqrt(np.sum(R**2,axis=0));
            
            
            '''calculate the vector potential Ae based on induced current'''
            phase=-k*r;
            r2=(k**2)*(r**2);
            r3=(k**3)*(r**3);
            '''1'''
            ee=np.exp(1j*phase)*k**2*(Je*(1j/phase-1/r2+1j/r3)+np.sum(Je*R/r,axis=0)*R/r*(-1j/phase+3/r2-3j/r3));
            Ee=np.sum(ee*N*ds,axis=1);
            '''2'''
            he=np.exp(1j*phase)*k**2
            he1=(R/r*1/(r2)*(1-1j*phase));
            he2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
            he2[0,...]=Je[1,...]*he1[2,...]-Je[2,...]*he1[1,...];
            he2[1,...]=Je[2,...]*he1[0,...]-Je[0,...]*he1[2,...];
            he2[2,...]=Je[0,...]*he1[1,...]-Je[1,...]*he1[0,...];
            He=np.sum(he*he2*N*ds,axis=1);
            '''3'''
            em=np.exp(1j*phase)*k**2
            em1=(R/r*1/r2*(1-1j*phase));
            em2=np.zeros((3,x1.size))+1j*np.zeros((3,x1.size));
            em2[0,...]=Jm[1,...]*em1[2,...]-Jm[2,...]*em1[1,...];
            em2[1,...]=Jm[2,...]*em1[0,...]-Jm[0,...]*em1[2,...];
            em2[2,...]=Jm[0,...]*em1[1,...]-Jm[1,...]*em1[0,...];
            Em=np.sum(em*em2*N*ds,axis=1);
            '''4'''
            hm=np.exp(1j*phase)*k**2*(Jm*(1j/phase-1/r2+1j/r3)+np.sum(Jm*R/r,axis=0)*R/r*(-1j/phase+3/r2-3j/r3));
            Hm=np.sum(hm*N*ds,axis=1);
            
            Field_E_x[i]=Z0/(4*np.pi)*Ee[0]-1/(4*np.pi)*Em[0];
            Field_E_y[i]=Z0/(4*np.pi)*Ee[1]-1/(4*np.pi)*Em[1];
            Field_E_z[i]=Z0/(4*np.pi)*Ee[2]-1/(4*np.pi)*Em[2];
        
            Field_H_x[i]=1/4/np.pi*He[0]+1/(4*np.pi*Z0)*Hm[0];
            Field_H_y[i]=1/4/np.pi*He[1]+1/(4*np.pi*Z0)*Hm[1];
            Field_H_z[i]=1/4/np.pi*He[2]+1/(4*np.pi*Z0)*Hm[2];
            #Field_H_x[i]=1/Z0*Field_E_x[i]
            #Field_H_y[i]=1/Z0*Field_E_y[i]
            #Field_H_z[i]=1/Z0*Field_E_z[i]
            
            
            
        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z;
    if Jm_in==0:
        Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus1(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                                              face1_n.N,face1_dS,JE);
    else:
        JM=np.append(np.append(Jm_in.x,Jm_in.y),Jm_in.z).reshape(3,-1);
        Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus2(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                                              face1_n.N,face1_dS,JE,JM);
    return Field_E,Field_H;


    
'''2.2 calculate the far-field beam'''    
def PO_far(face1,face1_n,face1_dS,face2,Field_in_E,Field_in_H,k,parallel=True):
   # output field:
    Field_E=vector();
    Field_H=vector();    
    Je_in=scalarproduct(2,crossproduct(face1_n,Field_in_H));
    if Field_in_E==0:
        Jm_in=0;
    else:
        Jm_in=scalarproduct(2,crossproduct(face1_n,Field_in_E));
    Je=np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,-1); 
    ''' calculate the field including the large matrix'''
    @njit(parallel=parallel)
    def calculus1(x1,y1,z1,x2,y2,z2,N,ds,Je): 
        Field_E_x=np.zeros(x2.size)+1j*np.zeros(x2.size)
        Field_E_y=np.zeros(x2.size)+1j*np.zeros(x2.size)
        Field_E_z=np.zeros(x2.size)+1j*np.zeros(x2.size)
        Field_H_x=np.zeros(x2.size)+1j*np.zeros(x2.size)
        Field_H_y=np.zeros(x2.size)+1j*np.zeros(x2.size)
        Field_H_z=np.zeros(x2.size)+1j*np.zeros(x2.size)
        r=np.zeros((3,1));
        rp=np.zeros((3,x1.size));
        rp[0,...]=x1;
        rp[1,...]=y1;
        rp[2,...]=z1;
        for i in prange(x2.size):
            r[0,0]=x2[i]
            r[1,0]=y2[i]
            r[2,0]=z2[i]
            '''calculate the vector potential Ae based on induced current'''
            phase=k*np.sum(r*rp,axis = 0)
            Ee=(k**2)*np.exp(1j*phase)*(Je-np.sum(Je*r,axis=0)*r);
            Ee=np.sum(Ee*N*ds,axis=1);
            He=(k**2)*np.exp(1j*phase)*Je

            He=np.sum(He*N*ds,axis=1);

            He=np.cross(r.ravel(),He);
            
            Field_E_x[i]=-1j*Z0/(4*np.pi)*Ee[0];
            Field_E_y[i]=-1j*Z0/(4*np.pi)*Ee[1];
            Field_E_z[i]=-1j*Z0/(4*np.pi)*Ee[2];

            Field_H_x[i]=-1j/4/np.pi*He[0];
            Field_H_y[i]=-1j/4/np.pi*He[1];
            Field_H_z[i]=-1j/4/np.pi*He[2];
        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z;
    '''Jm!=0'''
    @njit(parallel=parallel)
    def calculus2(x1,y1,z1,x2,y2,z2,N,ds,Je,Jm):
        Field_E_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_E_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_E_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        r=np.zeros((3,1))
        rp=np.zeros((3,x1.size))
        rp[0,...]=x1
        rp[1,...]=y1
        rp[2,...]=z1
        for i in prange(x2.size):
            r[0,0]=x2[i]
            r[1,0]=y2[i]
            r[2,0]=z2[i]
            
            '''calculate the vector potential Ae based on induced current'''
            phase=k*np.sum(r*rp,axis = 0)
            Ee=(k**2)*np.exp(1j*phase)*(Je-np.sum(Je*r,axis=0)*r);
            Ee=np.sum(Ee*N*ds,axis=1);
            #He=(k**2)*np.exp(1j*phase)*Je;
            #He=np.sum(He*N*ds,axis=1);
            #He=np.cross(r.T,He);
            
            Em=(k**2)*np.exp(1j*phase)*Jm;
            Em=np.sum(Em*N*ds,axis=1);
            Em=np.cross(r.T,Em);
            #Hm=(k**2)*np.exp(1j*phase)*(Jm-np.sum(Jm*r,axis=0)*r);
            #Hm=np.sum(Hm*N*ds,axis=1);
            
            Field_E_x[i]=-1j*Z0/(4*np.pi)*Ee[0]+1j/(4*np.pi)*Em[0];
            Field_E_y[i]=-1j*Z0/(4*np.pi)*Ee[1]+1j/(4*np.pi)*Em[1];
            Field_E_z[i]=-1j*Z0/(4*np.pi)*Ee[2]+1j/(4*np.pi)*Em[2];
        
            #Field_H_x[i]=-1j/4/np.pi*He[0]-1j/(4*np.pi*Z0)*Hm[0];
            #Field_H_y[i]=-1j/4/np.pi*He[1]-1j/(4*np.pi*Z0)*Hm[1];
            #Field_H_z[i]=-1j/4/np.pi*He[2]-1j/(4*np.pi*Z0)*Hm[2];
            H=1/Z0*np.cross(r,np.array([Field_E_x[i],Field_E_y[i],Field_E_z[i]]));
            Field_H_x[i]=H[1];
            Field_H_y[i]=H[2];
            Field_H_z[i]=H[3];
        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z;
    if Jm_in==0:
        Jm=0;
        Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus1(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                    face1_n.N,face1_dS,Je);
    else:
        Jm=np.append(np.append(Jm_in.x,Jm_in.y),Jm_in.z).reshape(3,-1);
        Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus2(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                    face1_n.N,face1_dS,Je,Jm);
    return Field_E,Field_H;
        

    
    
'''testing'''






def MATRIX(m1,m1_n,m1_dA,m2,Je_in,Jm_in,k):
    '''Field_in is current distribution'''
    Field_E=vector();
    Field_E.x=np.zeros(m2.x.shape,dtype=complex);
    Field_E.y=np.zeros(m2.x.shape,dtype=complex);
    Field_E.z=np.zeros(m2.x.shape,dtype=complex);
    Field_H=vector();
    Field_H.x=np.zeros(m2.x.shape,dtype=complex);
    Field_H.y=np.zeros(m2.x.shape,dtype=complex);
    Field_H.z=np.zeros(m2.x.shape,dtype=complex);
    Je=np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,-1);
    if Jm_in==0:
        Jm=0;
    else:
        Jm=np.append(np.append(Jm_in.x,Jm_in.y),Jm_in.z).reshape(3,-1);
        
    for i in range(m2.x.size):

        x=m2.x[i]-m1.x.reshape(1,-1);
        y=m2.y[i]-m1.y.reshape(1,-1);
        z=m2.z[i]-m1.z.reshape(1,-1);  
        R=np.append(np.append(x,y),z).reshape(3,-1);
        r=np.sqrt(x**2+y**2+z**2);
        del(x,y,z)
       
        ''' calculate the vector potential 'A_e' based on the induced current'''         
        phase=-k*r;
        r2=(k**2)*(r**2);
        r3=(k**3)*(r**3);
        
        Ee=np.exp(1j*phase)*k**2*(Je*(1j/phase-1/r2+1j/r3)+np.sum(Je*R/r,axis=0)*R/r*(-1j/phase+3/r2-3j/r3));
        Ee=np.sum(Ee*m1_n.N*m1_dA,axis=1);
        He=np.exp(1j*phase)*k**2*np.cross(Je.T,(R/r*1/(r2)*(1-1j*phase)).T).T;
        He=np.sum(He*m1_n.N*m1_dA,axis=1);  
        if Jm_in==0:
            Em=np.zeros(3);
            Hm=np.zeros(3);
            Field_E.x[i]=Z0/(4*np.pi)*Ee[0]
            Field_E.y[i]=Z0/(4*np.pi)*Ee[1]
            Field_E.z[i]=Z0/(4*np.pi)*Ee[2]
        
            Field_H.x[i]=1/4/np.pi*He[0]
            Field_H.y[i]=1/4/np.pi*He[1]
            Field_H.z[i]=1/4/np.pi*He[2]
        else:
            Em=np.exp(1j*phase)*k**2*np.cross(Jm.T,(R/r*1/r2*(1-1j*phase)).T).T;
            Em=np.sum(Em*m1_n.N*m1_dA,axis=1);
            Hm=np.exp(1j*phase)*k**2*(Jm*(1j/phase-1/r2+1j/r3)+np.sum(Jm*R/r,axis=0)*R/r*(-1j/phase+3/r2-3j/r3));
            Hm=np.sum(Hm*m1_n.N*m1_dA,axis=1);
            
            Field_E.x[i]=Z0/(4*np.pi)*Ee[0]-1/(4*np.pi)*Em[0];
            Field_E.y[i]=Z0/(4*np.pi)*Ee[1]-1/(4*np.pi)*Em[1];
            Field_E.z[i]=Z0/(4*np.pi)*Ee[2]-1/(4*np.pi)*Em[2];
        
            Field_H.x[i]=1/4/np.pi*He[0]+1/(4*np.pi*Z0)*Hm[0];
            Field_H.y[i]=1/4/np.pi*He[1]+1/(4*np.pi*Z0)*Hm[1];
            Field_H.z[i]=1/4/np.pi*He[2]+1/(4*np.pi*Z0)*Hm[2];
    
    
    return Field_E,Field_H;