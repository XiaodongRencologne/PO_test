#!/usr/bin/env python
# coding: utf-8

# In[1]:

from tqdm import tqdm
import numpy as np;
import torch as T;
from numba import njit, prange;
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

def PO_GPU(face1,face1_n,face1_dS,face2,Field_in_E,Field_in_H,k,device =T.device('cuda')):
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
    face1_dS =T.tensor(face1_dS).to(device)
    
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
        ee=T.exp(1j*phase)*k**2*(Je*(1j/phase-1/r2-1j/r3)+T.sum(Je*R/r,axis=0)*R/r*(-1j/phase+3/r2+3j/r3))
        Ee=T.sum(ee*face1_n.N*face1_dS,axis=-1)
        del(ee)
        '''2'''
        he=T.exp(1j*phase)*k**2
        he1=(R*1/(r2)*(1-1j*phase))
        he2 = T.zeros((3,N_points, N_current),dtype=T.complex128).to(device)

        '''
        he2[0,...]=Je[1,...]*he1[2,...]-Je[2,...]*he1[1,...]
        he2[1,...]=Je[2,...]*he1[0,...]-Je[0,...]*he1[2,...]
        he2[2,...]=Je[0,...]*he1[1,...]-Je[1,...]*he1[0,...]
        '''

        he2 = T.cross(Je, he1,dim=0)


        
        He=T.sum(he*he2*face1_n.N*face1_dS,axis=-1)

        F_E_x=1/(4*np.pi)*Ee[0,...]
        F_E_y=1/(4*np.pi)*Ee[1,...]
        F_E_z=1/(4*np.pi)*Ee[2,...]
        
        F_H_x=1/4/np.pi*He[0,...]/Z0
        F_H_y=1/4/np.pi*He[1,...]/Z0
        F_H_z=1/4/np.pi*He[2,...]/Z0
        return F_E_x,F_E_y,F_E_z,F_H_x,F_H_y,F_H_z
    if device==T.device('cuda'):
        M_all=T.cuda.get_device_properties(0).total_memory
        M_element=Je_in.x.itemsize * Je_in.x.size * 4
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
    
'''2.2 calculate the far-field beam'''    
def PO_far(face1,face1_n,face1_dS,face2,Field_in_E,Field_in_H,k,parallel=True):
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
            Ee = (JE-np.sum(JE*r,axis = 0)*r)* np.exp(1j*phase)*k**2
            Ee = np.sum(Ee*N,axis=-1)*ds*(-1j*Z0/4/np.pi)
            
            Field_E_x[i] = Ee[0]
            Field_E_y[i] = Ee[1]
            Field_E_z[i] = Ee[2]
        return Field_E_x,Field_E_y,Field_E_z,Field_H_x,Field_H_y,Field_H_z
    Field_E.x,Field_E.y,Field_E.z,Field_H.x,Field_H.y,Field_H.z=calculus1(face1.x,face1.y,face1.z,face2.x,face2.y,face2.z,
                                                    face1_n.N,face1_dS)
    return Field_E,Field_H
        

def PO_far_GPU(face1,face1_n,face1_dS,face2,Field_in_E,Field_in_H,k,device =T.device('cuda')):
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
    JE=T.tensor(np.append(np.append(Je_in.x,Je_in.y),Je_in.z).reshape(3,1,-1),dtype = T.complex128).to(device)

    face1.np2Tensor(device)
    N_current = face1.x.size()[0]
    face1_n.np2Tensor(device)
    face2.np2Tensor(device)
    rp = T.tensor((3,1,face1.x.size()[0]))
    rp[0,...] = face1.x.reshape((1,-1))
    rp[1,...] = face1.y.reshape((1,-1))
    rp[2,...] = face1.z.reshape((1,-1))

    def calcu(x2,y2,z2,Je):
        N_points = x2.size()[0]
        #print(N_points)
        r = T.zeros((3,N_points,1)).to(device)
        r[0,:,:] = x2.reshape(-1,1) 
        r[1,:,:] = y2.reshape(-1,1)
        r[2,:,:] = z2.reshape(-1,1)
        phase = k*T.sum(rp*r,axis = 0)
        Ee = (JE-T.sum(JE*r,axis = 0)*r)* np.exp(1j*phase)*k**2
        Ee = np.sum(Ee*face1_n.N,axis=-1)*face1_dS*(-1j*Z0/4/np.pi)

        F_E_x = Ee[0,...]
        F_E_y = Ee[1,...]
        F_E_z = Ee[2,...]

        F_H_x = 0
        F_H_y = 0
        F_H_z = 0
        return F_E_x,F_E_y,F_E_z,F_H_x,F_H_y,F_H_z
    
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