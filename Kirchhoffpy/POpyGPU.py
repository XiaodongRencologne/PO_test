#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
from tqdm import tqdm
import numpy as np
import torch as T
from torch.cuda.amp import autocast, GradScaler
cpu_cores = T.get_num_threads()
print(cpu_cores)
T.set_num_threads(cpu_cores*2)
from numba import njit, prange
from .Vopy import vector,crossproduct,scalarproduct,abs_v,dotproduct,sumvector

import copy;
import time;
c=299792458;
mu=4*np.pi*10**(-7);
epsilon=8.854187817*10**(-12);
Z0=np.sqrt(mu/epsilon,dtype = np.float64)
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
        if type(self.x).__module__==T.__name__:
            self.x=self.x.cpu().numpy() 
        if type(self.y).__module__==T.__name__:
            self.y=self.y.cpu().numpy()
        if type(self.z).__module__==T.__name__:
            self.z=self.z.cpu().numpy()
        else:
            pass;
        

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
        Field_real=np.zeros(x2.size);
        Field_imag=np.zeros(x2.size);
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
        Matrix.real,Matrix.imag,Field_face2.real,Field_face2.imag,COS_R=calculus1(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
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
        Field_real=np.zeros(x2.size);
        Field_imag=np.zeros(x2.size);
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
        Matrix.real,Matrix.imag,Field_face2.real,Field_face2.imag,COS_R=calculus1(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
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
def PO(face1,face1_n,face1_dS,
    face2,
    Field_in_E,Field_in_H,
    k,parallel=True):
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
"""
def PO_GPU_0v(face1,face1_n,face1_dS,face2,Field_in_E,Field_in_H,k,n,device =T.device('cuda')):
    # output field:
    N_f = face2.x.size
    Field_E=vector()
    Field_E.x = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_E.y = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_E.z = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_H=vector()
    Field_H.x = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_H.y = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_H.z = np.zeros(N_f) + 1j*np.zeros(N_f)
    # input field converted to surface currents
    Je_in=scalarproduct(2,crossproduct(face1_n,Field_in_H))
    JE=T.tensor(np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,1,-1)).to(device)
    #del(Je_in)
    #print(face1_n.z.reshape(101,-1))
    #print(Field_in_H.y.reshape(101,-1))
    #print('Je:', JE[0,:].reshape(101,-1))
    face1.np2Tensor(device)
    N_current = face1.x.size()[0]
    face1_n.np2Tensor(device)
    face2.np2Tensor(device)
    face1_dS =T.tensor(face1_dS,dtype = T.float64).to(device)
    k = k*n
    Z = Z0/n/Z0
    def calcu(x2,y2,z2,Je):
        N_points = x2.size()[0]
        #print(N_points)
        R = T.zeros((3,N_points,N_current),dtype = T.float64).to(device)
        R[0,:,:] = x2.reshape(-1,1) - face1.x.ravel()
        R[1,:,:] = y2.reshape(-1,1) - face1.y.ravel()
        R[2,:,:] = z2.reshape(-1,1) - face1.z.ravel()
        r = T.sqrt(T.sum(R**2,axis=0))
        #R =R/r # R is the normlized vector
        phase = -k*r
        r2 = phase**2
        r3 = phase**3
        '''1'''
        factor1=T.exp(1j*phase)*k*k
        ee=factor1*(Je*(1j/phase-1/r2-1j/r3)+T.sum(Je*R/r,axis=0)*R/r*(-1j/phase+3/r2+3j/r3))
        Ee=Z/(4*T.pi)*T.sum(ee*face1_n.N*face1_dS,axis=-1)
        del(ee)
        '''2'''
        
        #he1=(R*1/(r2)*(1-1j*phase))
        #he2 = T.zeros((3,N_points, N_current),dtype=T.complex128).to(device)

        '''
        he2[0,...]=Je[1,...]*he1[2,...]-Je[2,...]*he1[1,...]
        he2[1,...]=Je[2,...]*he1[0,...]-Je[0,...]*he1[2,...]
        he2[2,...]=Je[0,...]*he1[1,...]-Je[1,...]*he1[0,...]
        '''
        #he2 = T.cross(Je, R.type(T.cdouble),dim=0)*1/(r2)*(1-1j*phase)
        he2 = T.cross(Je, (R/r).type(T.cdouble),dim=0)*1/(r2)*(1-1j*phase)
        #del(he1)
        He=T.sum(factor1*he2*face1_n.N*face1_dS,axis=-1)/(4*T.pi)

        F_E_x=Ee[0,...]
        F_E_y=Ee[1,...]
        F_E_z=Ee[2,...]
        
        F_H_x=He[0,...]
        F_H_y=He[1,...]
        F_H_z=He[2,...]
        return F_E_x,F_E_y,F_E_z,F_H_x,F_H_y,F_H_z
    if device==T.device('cuda'):
        M_all=T.cuda.get_device_properties(0).total_memory
        M_element=Je_in.x.itemsize * Je_in.x.size * 5
        cores=int(M_all/M_element/6)
        print('cores:',cores)
    else:
        cores=os.cpu_count()*20
        print('cores:',cores)
    N=face2.x.nelement()
    Ni = int(N/cores)
    for i in tqdm(prange(Ni)):
        E_X,E_Y,E_Z,H_X,H_Y,H_Z=calcu(face2.x[i*cores:(i+1)*cores],
                                      face2.y[i*cores:(i+1)*cores],
                                      face2.z[i*cores:(i+1)*cores],
                                      JE)
        Field_E.x[i*cores:(i+1)*cores] = E_X.cpu().numpy()
        Field_E.y[i*cores:(i+1)*cores] = E_Y.cpu().numpy()
        Field_E.z[i*cores:(i+1)*cores] = E_Z.cpu().numpy()
        Field_H.x[i*cores:(i+1)*cores] = H_X.cpu().numpy()
        Field_H.y[i*cores:(i+1)*cores] = H_Y.cpu().numpy()
        Field_H.z[i*cores:(i+1)*cores] = H_Z.cpu().numpy()
    
    if int(N%cores)!=0:
        E_X,E_Y,E_Z,H_X,H_Y,H_Z=calcu(face2.x[Ni*cores:],
                                      face2.y[Ni*cores:],
                                      face2.z[Ni*cores:],
                                      JE)
        Field_E.x[Ni*cores:] = E_X.cpu().numpy()
        Field_E.y[Ni*cores:] = E_Y.cpu().numpy()
        Field_E.z[Ni*cores:] = E_Z.cpu().numpy()
        Field_H.x[Ni*cores:] = H_X.cpu().numpy()
        Field_H.y[Ni*cores:] = H_Y.cpu().numpy()
        Field_H.z[Ni*cores:] = H_Z.cpu().numpy()
    face1.Tensor2np()
    face1_n.Tensor2np()
    face2.Tensor2np()
    T.cuda.empty_cache()
    return Field_E,Field_H
"""

def PO_GPU(face1,face1_n,face1_dS,
           face2,
           Field_in_E,Field_in_H,
           k,n,
           device =T.device('cuda')):
    """
    Optimized Physical Optics (PO) GPU implementation for near-field calculations.

    Parameters:
        face1, face1_n, face1_dS: Surface 1 geometry, normals, and differential area.
        face2: Surface 2 geometry.
        Field_in_E, Field_in_H: Incident electric and magnetic fields.
        k: Wave number.
        n: Refractive index.
        device: PyTorch device (default: CUDA).

    Returns:
        Field_E, Field_H: Resultant electric and magnetic fields on face2.
    """

    # Set data types based on precision
    real_dtype = T.float64
    complex_dtype = T.complex128

    # Initialize output fields on GPU
    N_f = face2.x.size
    Field_E = vector()
    Field_E.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.z = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H = vector()
    Field_H.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.z = T.zeros(N_f, dtype=T.complex128, device=device)

    # Convert input fields to surface currents
    Je_in = scalarproduct(2, crossproduct(face1_n, Field_in_H))
    JE = T.tensor(np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,1,-1), 
                  dtype=T.complex128,
                  device=device)

    # Move face1 and face2 data to GPU
    face1.np2Tensor(device)
    face1_n.np2Tensor(device)
    face2.np2Tensor(device)
    face1_dS = T.tensor(face1_dS, dtype=real_dtype, device=device)
    ds = face1_n.N * face1_dS
    print(ds.device)
    k = T.tensor(k*n,device = device)
    Z = T.tensor(Z0 / n/ Z0,device = device)
    pi = T.tensor(4*T.pi, device = device)
    J = T.tensor(0+1j, device = device)

    Surf1 = T.stack([face1.x.ravel(), 
                      face1.y.ravel(), 
                      face1.z.ravel()], dim=0).reshape(3,1,-1).to(device)
    Surf2 = T.stack([face2.x.ravel(), 
                      face2.y.ravel(), 
                      face2.z.ravel()], dim=0).reshape(3,-1,1).to(device)
    print(JE.device, Surf1.device, Surf2.device, k.device, Z.device, pi.device)
    

    #@T.jit.script
    def calculate_fields(s2,surf1,Je,JJ,K,z,PI,dS):
        """
        Helper function to calculate fields for a batch of points.
        """
        # Compute R and r
        R = s2 - surf1
        r = T.linalg.norm(R, dim=0) # Compute the norm directly
        R_n = R / r  # Normalize the vector
        del(R)
        
        # Compute phase and amplitude terms
        r_inv = r.clone()  # Avoid modifying `r` directly
        r_inv.reciprocal_().mul_(-1 / K)  # In-place reciprocal and scaling
        r2_inv = r_inv.clone().mul_(r_inv)  # In-place square
        r3_inv = r2_inv.clone().mul_(r_inv)  # In-place cube

        
        factor1=(T.exp((-JJ * K) * r) * K**2)
        # Magnetic field calculation
        He=T.sum( (factor1 * r2_inv * (1 + (JJ * K) * r ) * dS) * T.cross(Je, (R_n).type(T.complex128) , dim=0), dim=-1) / PI

        # Electric field calculation
        #sum_result = T.sum(Je * R_n, dim = 0).to(device)
        ee = factor1 * (
            Je*(JJ* (r_inv - r3_inv) -  r2_inv )
            +  T.sum(Je * R_n, dim = 0) * ( 3 * r2_inv + JJ * (3*r3_inv - r_inv )) * R_n
        )
        #del(sum_result)
        Ee=z / PI * T.sum(ee * dS , dim=-1)
        del(ee)


        return Ee, He
    # Determine batch size based on available memory
    if device == T.device('cuda'):
        # Get total, allocated, and reserved memory
        total_memory = T.cuda.get_device_properties(0).total_memory
        allocated_memory = T.cuda.memory_allocated(0)
        reserved_memory = T.cuda.memory_reserved(0)

        # Calculate free memory
        free_memory = total_memory - reserved_memory

        # Adjust batch size based on free memory
        element_size = JE.element_size() * JE.nelement()
        batch_size = int(free_memory / element_size / 8)  # Reduce divisor for smaller batches
        
    else:
        batch_size = os.cpu_count() * 10
        T.set_num_threads(os.cpu_count())
    """
    if device == T.device('cuda'):
        total_memory = T.cuda.get_device_properties(0).total_memory
        element_size = JE.element_size() * JE.nelement()
        batch_size = int(total_memory / element_size / 6)
    else:
        batch_size = os.cpu_count() * 20
    """
    print(f"Batch size: {batch_size}")
    N = face2.x.nelement()
    num_batches = N // batch_size

    # Process batches
    
    #with T.no_grad():
    N_current = face1.x.size
    for i in tqdm(range(num_batches),mininterval=5):
        start = i * batch_size
        end = (i + 1) * batch_size
        #idx = T.arange(start, end, device ='cuda')
        Ee, He = calculate_fields(Surf2[:,start:end,:],Surf1,JE,J,k,Z,pi,ds)

        Field_E.x[start:end] = Ee[0, :]
        Field_E.y[start:end] = Ee[1, :]
        Field_E.z[start:end] = Ee[2, :]
        Field_H.x[start:end] = He[0, :]
        Field_H.y[start:end] = He[1, :]
        Field_H.z[start:end] = He[2, :]

        '''
        if i % 100 == 0:
            T.cuda.empty_cache()  # Frees unused memory
            T.cuda.synchronize()
        '''
            
    # Process remaining points
    if N % batch_size != 0:
        start = num_batches * batch_size
        Ee, He = calculate_fields(Surf2[:,start:,:],Surf1,JE,J,k,Z,pi,ds)
        Field_E.x[start:] = Ee[0, :]
        Field_E.y[start:] = Ee[1, :]
        Field_E.z[start:] = Ee[2, :]
        Field_H.x[start:] = He[0, :]
        Field_H.y[start:] = He[1, :]
        Field_H.z[start:] = He[2, :]

    # Move tensors back to CPU only once    
    Field_E.Tensor2np()
    Field_H.Tensor2np()
    face1.Tensor2np()
    face1_n.Tensor2np()
    face2.Tensor2np()
    T.cuda.empty_cache()
    T.cuda.synchronize()
    return Field_E, Field_H



def PO_GPU_2(face1,face1_n,face1_dS,
           face2,
           Field_in_E,Field_in_H,
           k,n,
           device =T.device('cuda')):
    """
    Optimized Physical Optics (PO) GPU implementation for near-field calculations.

    Parameters:
        face1, face1_n, face1_dS: Surface 1 geometry, normals, and differential area.
        face2: Surface 2 geometry.
        Field_in_E, Field_in_H: Incident electric and magnetic fields.
        k: Wave number.
        n: Refractive index.
        device: PyTorch device (default: CUDA).

    Returns:
        Field_E, Field_H: Resultant electric and magnetic fields on face2.
    """

    # Set data types based on precision
    real_dtype = T.float64

    # Initialize output fields on GPU
    N_f = face2.x.size
    Field_E = vector()
    Field_E.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.z = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H = vector()
    Field_H.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.z = T.zeros(N_f, dtype=T.complex128, device=device)

    k_n = k * n
    k0 = k * 1.0

    ds = face1_dS*face1_n.N * k_n**2 /4/np.pi

    # constent Number 
    k_n = T.tensor(k_n,device = device)
    k0 = T.tensor(k0,device = device)
    Z = T.tensor(Z0 / n/ Z0,device = device)

    # Convert input fields to surface currents
    start_time = time.time()
    J_in = scalarproduct(2*ds, crossproduct(face1_n, Field_in_H))
    JE = T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1), 
                  dtype=T.complex128,
                  device=device).contiguous()
    
    J_in = scalarproduct(2*ds, crossproduct(face1_n, Field_in_E))
    JM = T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1), 
                  dtype=T.complex128,
                  device=device).contiguous()

    #print('tiemusage:',time.time() - start_time)

    # convert surface points into tensor
    Surf1 = T.stack([T.tensor(face1.x.ravel()), 
                     T.tensor(face1.y.ravel()), 
                     T.tensor(face1.z.ravel())], dim=0).reshape(3,1,-1)
    Surf1 = Surf1.contiguous().to(device)
    Surf2 = T.stack([T.tensor(face2.x.ravel()), 
                     T.tensor(face2.y.ravel()), 
                     T.tensor(face2.z.ravel())], dim=0).reshape(3,-1,1).to(device)
    Surf2 = Surf2.contiguous().to(device)

    #N_current = face1.x.size
    #N_target = face2.x.size 
    #R_n_cpu = T.zeros((3,N_target,N_current),dtype = T.complex128,device = 'cpu', pin_memory=True)

    #Memory_size = R_n_cpu.element_size() * R_n_cpu.nelement()


    #@T.jit.script
    '''
    def calculate_R(point1,point2,K,R_n):
        R = point2 - point1
        r = T.linalg.norm(R, dim=0) # Compute the norm directly
        R = R / r  # Normalize the vector
        r.mul_(K)
        Factor = T.exp(-1j*r)*(1+1j*r)/r**2
        R_n = R * Factor
    '''
    
    def calculate_fields(s2,K):
        """
        Helper function to calculate fields for a batch of points.
        """
        
        # Compute R and r
        R = (s2 - Surf1).contiguous()
        r = T.linalg.norm(R, dim=0).contiguous().unsqueeze(0) # Compute the norm directly
        R = R / r # Normalize the vector
        r = r * K
        R = R * T.exp(-1j*r)*(1+1j*r)/r**2
        del(r,s2)
        # Magnetic field calculation
        he = T.cross(JE, R, dim=0).contiguous()
        He = he.sum(dim=-1)
        del(he)

        # Electric field calculation
        ee = T.cross(JM, R, dim = 0).contiguous()
        Ee = ee.sum(dim = -1)
        del(ee)
        return Ee, He
    # Determine batch size based on available memory
    if device == T.device('cuda'):
        # Get total, allocated, and reserved memory
        total_memory = T.cuda.get_device_properties(0).total_memory
        allocated_memory = T.cuda.memory_allocated(0)
        reserved_memory = T.cuda.memory_reserved(0)

        # Calculate free memory
        free_memory = total_memory - reserved_memory

        # Adjust batch size based on free memory
        element_size = JE.element_size() * JE.nelement()
        batch_size = int(free_memory / element_size / 20)
    else:
        batch_size = os.cpu_count() * 10

    print(f"Batch size: {batch_size}")
    N = face2.x.size
    num_batches = N // batch_size

    # Process batches
    #print(Surf2.device, Surf1.device, k_n.device)
    #print(Surf2.is_contiguous(), Surf1.is_contiguous(), k_n.is_contiguous())

    with autocast():
        with T.no_grad():
            for i in tqdm(prange(num_batches),mininterval=5):
                start = i * batch_size
                end = (i + 1) * batch_size
                #idx = T.arange(start, end, device ='cuda')
                Ee, He = calculate_fields(Surf2[:,start:end,:].contiguous() ,k_n)

                Field_E.x[start:end] = Ee[0, :]
                Field_E.y[start:end] = Ee[1, :]
                Field_E.z[start:end] = Ee[2, :]
                Field_H.x[start:end] = He[0, :]
                Field_H.y[start:end] = He[1, :]
                Field_H.z[start:end] = He[2, :]
                    
            # Process remaining points
            if N % batch_size != 0:
                start = num_batches * batch_size
                Ee, He = calculate_fields(Surf2[:,start:,:].contiguous(),k_n)
                Field_E.x[start:] = Ee[0, :]
                Field_E.y[start:] = Ee[1, :]
                Field_E.z[start:] = Ee[2, :]
                Field_H.x[start:] = He[0, :]
                Field_H.y[start:] = He[1, :]
                Field_H.z[start:] = He[2, :]
    # Move tensors back to CPU only once    
    Field_E.Tensor2np()
    Field_H.Tensor2np()

    T.cuda.empty_cache()
    T.cuda.synchronize()
    return Field_E, Field_H


'''2.2 calculate the far-field beam'''    
def PO_far(face1,face1_n,face1_dS,face2,Field_in_E,Field_in_H,k,parallel=True,device =T.device('cpu')):
   # output field:
    Field_E=vector()
    Field_H=vector()  
    Je_in=scalarproduct(2,crossproduct(face1_n,Field_in_H))
    JE=np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,-1)
    print(JE.shape)
    ''' calculate the field including the large matrix'''
    #@njit(parallel=parallel)
    def calculus1(x1,y1,z1,x2,y2,z2,N,ds): 
        Field_E_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_E_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_E_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_x=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_y=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        Field_H_z=np.zeros(x2.shape)+1j*np.zeros(x2.shape)
        rp=np.zeros((3,x1.size))
        rp[0,:] = x1
        rp[1,:] = y1
        rp[2,:] = z1
        for i in prange(x2.size):
            r = np.array([[x2[i]],[y2[i]],[z2[i]]])
            phase = k*np.sum(rp*r,axis=0)
            Ee = (JE-np.sum(JE*r,axis = 0) * r) * np.exp(1j*phase)*k**2
            Ee = np.sum(Ee*N*ds,axis=-1)*(-1j*Z0/4/np.pi)
            #print(Ee)
            Field_E_x[i] = Ee[0]
            Field_E_y[i] = Ee[1]
            Field_E_z[i] = Ee[2]
        #print(Field_E_x)
        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z
    Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus1(face1.x,face1.y,face1.z,
                                                                          face2.x.ravel(),face2.y.ravel(),face2.z.ravel(),
                                                                          face1_n.N.ravel(),face1_dS.ravel())
    #print(Field_E.x)
    return Field_E,Field_H
        

def PO_far_GPU(face1,face1_n,face1_dS,
               face2,
               Field_in_E,
               Field_in_H,
               k,n=1,
               device =T.device('cuda')):
    # output field:
    N_f = face2.x.size
    Field_E=vector()
    Field_E.x = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_E.y = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_E.z = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_H=vector()
    Field_H.x = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_H.y = np.zeros(N_f) + 1j*np.zeros(N_f)
    Field_H.z = np.zeros(N_f) + 1j*np.zeros(N_f)

    Je_in=scalarproduct(2,crossproduct(face1_n,Field_in_H))
    JE=T.tensor(np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,1,-1),
                dtype = T.complex128).to(device)

    face1.np2Tensor(device)
    N_current = face1.x.size()[0]
    face1_n.np2Tensor(device)
    face2.np2Tensor(device)
    face2.x = face2.x.ravel()
    face2.y = face2.y.ravel()
    face2.z = face2.z.ravel()
    rp = T.zeros((3,1,N_current),dtype = T.float64).to(device)
    rp[0,...] = face1.x.reshape((1,-1))
    rp[1,...] = face1.y.reshape((1,-1))
    rp[2,...] = face1.z.reshape((1,-1))
    face1_dS =T.tensor(face1_dS).to(device)
    
    def calcu(x2,y2,z2):
        N_points = x2.size()[0]
        #print(N_points)
        r = T.zeros((3,N_points,1)).to(device)
        r[0,:,:] = x2.reshape(-1,1) 
        r[1,:,:] = y2.reshape(-1,1)
        r[2,:,:] = z2.reshape(-1,1)
        phase = k*T.sum(rp*r,axis = 0)
        Ee = (JE-T.sum(JE*r,axis = 0)*r)* T.exp(1j*phase)*k**2
        Ee = T.sum(Ee*face1_n.N*face1_dS,axis=-1)*(-1j*Z0/4/T.pi)/Z0

        F_E_x = Ee[0,...]
        F_E_y = Ee[1,...]
        F_E_z = Ee[2,...]
        # calculate Magnetic field
        He = T.cross(r.reshape(3,-1).type(T.cdouble),Ee,axis=0)/Z0
        F_H_x = He[0,...]
        F_H_y = He[1,...]
        F_H_z = He[2,...]
        return F_E_x, F_E_y, F_E_z, F_H_x, F_H_y, F_H_z
    
    if device==T.device('cuda'):
        M_all=T.cuda.get_device_properties(0).total_memory
        M_element=Je_in.x.itemsize * Je_in.x.size * 3
        cores=int(M_all/M_element/6)
        print('cores:',cores)
    else:
        cores=os.cpu_count()*20
        print('cores:',cores)
    N=face2.x.nelement()
    Ni = int(N/cores)
    for i in tqdm(range(Ni)):
        E_X,E_Y,E_Z,H_X,H_Y,H_Z=calcu(face2.x[i*cores:(i+1)*cores],
                                      face2.y[i*cores:(i+1)*cores],
                                      face2.z[i*cores:(i+1)*cores])
        Field_E.x[i*cores:(i+1)*cores] = E_X.cpu().numpy()
        Field_E.y[i*cores:(i+1)*cores] = E_Y.cpu().numpy()
        Field_E.z[i*cores:(i+1)*cores] = E_Z.cpu().numpy()
        Field_H.x[i*cores:(i+1)*cores] = H_X.cpu().numpy()
        Field_H.y[i*cores:(i+1)*cores] = H_Y.cpu().numpy()
        Field_H.z[i*cores:(i+1)*cores] = H_Z.cpu().numpy()
    
    if int(N%cores)!=0:
        E_X,E_Y,E_Z,H_X,H_Y,H_Z=calcu(face2.x[Ni*cores:],
                                      face2.y[Ni*cores:],
                                      face2.z[Ni*cores:])
        Field_E.x[Ni*cores:] = E_X.cpu().numpy()
        Field_E.y[Ni*cores:] = E_Y.cpu().numpy()
        Field_E.z[Ni*cores:] = E_Z.cpu().numpy()
        Field_H.x[Ni*cores:] = H_X.cpu().numpy()
        Field_H.y[Ni*cores:] = H_Y.cpu().numpy()
        Field_H.z[Ni*cores:] = H_Z.cpu().numpy()
    face1.Tensor2np()
    face1_n.Tensor2np()
    face2.Tensor2np()
    T.cuda.empty_cache()  
    T.cuda.synchronize()  
    return Field_E,Field_H   
    

def PO_far_GPU2(face1,face1_n,face1_dS,
               face2,
               Field_in_E,
               Field_in_H,
               k,
               device =T.device('cuda')):
    # output field:
    N_f = face2.x.size
    Field_E = vector()
    Field_E.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_E.z = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H = vector()
    Field_H.x = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.y = T.zeros(N_f, dtype=T.complex128, device=device)
    Field_H.z = T.zeros(N_f, dtype=T.complex128, device=device)

    ds = face1_dS * face1_n.N * k**2 /4/np.pi


    J_in=scalarproduct(2*ds,crossproduct(face1_n,Field_in_H))
    JE=T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1),
                dtype = T.complex128).to(device)
    
    J_in = scalarproduct(2*ds, crossproduct(face1_n, Field_in_E))
    JM = T.tensor(np.append(np.append(J_in.x,J_in.y),J_in.z).reshape(3,1,-1), 
                  dtype=T.complex128,
                  device=device).contiguous()
    
    # convert surface points into tensor
    Surf1 = T.stack([T.tensor(face1.x.ravel()), 
                     T.tensor(face1.y.ravel()), 
                     T.tensor(face1.z.ravel())], dim=0).reshape(3,1,-1)
    Surf1 = Surf1.contiguous().to(device)
    Surf2 = T.stack([T.tensor(face2.x.ravel()), 
                     T.tensor(face2.y.ravel()), 
                     T.tensor(face2.z.ravel())], dim=0).reshape(3,-1,1)
    Surf2 = Surf2.contiguous().to(device)

    def calculate_fields(s2,K):
        """
        Helper function to calculate fields for a batch of points.
        """
        # Compute R and r
        Phase = k * T.sum((s2 * Surf1).contiguous(),axis = 0).contiguous()
        Phase =  T.exp(1j * Phase) #* ds
        # Electric field calculation
        ee = T.sum( JM * Phase, axis = -1 ).contiguous()
        r = s2.reshape(3,-1).contiguous().to(dtype = T.complex128)
        Ee = 1j * T.cross(r, ee, dim=0).contiguous()
        # Magnetic field calculation
        he = T.sum( JE * Phase, axis = -1 ).contiguous()
        He = -1j /Z0 * T.cross(r, he, dim=0).contiguous()

        return Ee, He
    
    if device == T.device('cuda'):
        # Get total, allocated, and reserved memory
        total_memory = T.cuda.get_device_properties(0).total_memory
        allocated_memory = T.cuda.memory_allocated(0)
        reserved_memory = T.cuda.memory_reserved(0)

        # Calculate free memory
        free_memory = total_memory - reserved_memory

        # Adjust batch size based on free memory
        element_size = JE.element_size() * JE.nelement()
        batch_size = int(free_memory / element_size / 6)
    else:
        batch_size = os.cpu_count() * 30

    print(f"Batch size: {batch_size}")
    N = face2.x.size
    num_batches = N // batch_size

    # Process batches
    #print(Surf2.device, Surf1.device, k_n.device)
    #print(Surf2.is_contiguous(), Surf1.is_contiguous(), k_n.is_contiguous())

    with T.no_grad():
        for i in tqdm(range(num_batches),mininterval=5):
            start = i * batch_size
            end = (i + 1) * batch_size
            #idx = T.arange(start, end, device ='cuda')
            Ee, He = calculate_fields(Surf2[:,start:end,:].contiguous() ,k)

            Field_E.x[start:end] = Ee[0, :]
            Field_E.y[start:end] = Ee[1, :]
            Field_E.z[start:end] = Ee[2, :]
            Field_H.x[start:end] = He[0, :]
            Field_H.y[start:end] = He[1, :]
            Field_H.z[start:end] = He[2, :]
                
        # Process remaining points
        if N % batch_size != 0:
            start = num_batches * batch_size
            Ee, He = calculate_fields(Surf2[:,start:,:].contiguous(),k)
            Field_E.x[start:] = Ee[0, :]
            Field_E.y[start:] = Ee[1, :]
            Field_E.z[start:] = Ee[2, :]
            Field_H.x[start:] = He[0, :]
            Field_H.y[start:] = He[1, :]
            Field_H.z[start:] = He[2, :]
    # Move tensors back to CPU only once    
    Field_E.Tensor2np()
    Field_H.Tensor2np()

    T.cuda.empty_cache()
    T.cuda.synchronize()
    return Field_E, Field_H
 

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