#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
This package provides N input beams, and each beam function can offer scalar and vector modes.
1. Gaussian beam in far field;
2. Gaussian beam near field;
'''

import numpy as np;
from .coordinate_operations import cartesian_to_spherical as cart2spher;
from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;



# In[5]:
class vector:
    def __init__(self):
        self.x=np.array([]);
        self.y=np.array([]);
        self.z=np.array([]);
        
def dotproduct(A,B):
    return A.x.ravel()*B.x.ravel()+A.y.ravel()*B.y.ravel()+A.z.ravel()*B.z.ravel();

def scalarproduct(KK,A):
    B=vector();
    B.x=KK*A.x;
    B.y=KK*A.y;
    B.z=KK*A.z;    
    return B
def sumvector(A,B):
    C=vector();
    C.x=A.x+B.x;
    C.y=A.y+B.y;
    C.z=A.z+B.z;
    return C;
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

'''
Type 1: Gaussian beam;
'''

def Gaussibeam(Edge_taper,Edge_angle,k,Mirror_in,Mirror_n,angle,displacement,polarization='scalar'):
    '''
    param 1: 'Edge_taper' define ratio of maximum power and the edge power in the antenna;
    param 2: 'Edge_angle' is the angular size of the mirror seen from the feed coordinates;
    param 3: 'k' wave number;
    param 4: 'Mirror_in' the sampling points in the mirror illumanited by feed;
    param 5: 'fieldtype' chose the scalar mode or vector input field.
    '''
    
    Mirror_in=global2local(angle,displacement,Mirror_in);
    Mirror_n=global2local(angle,[0,0,0],Mirror_n);
    if polarization.lower()=='scalar':
        Theta_max=Edge_angle;
        E_taper=Edge_taper;
        b=(20*np.log10((1+np.cos(Theta_max))/2)-E_taper)/(20*k*(1-np.cos(Theta_max))*np.log10(np.exp(1)));
        w0=np.sqrt(2/k*b)
        r,theta,phi=cart2spher(Mirror_in.x,Mirror_in.y,Mirror_in.z);
        R=np.sqrt(r**2-b**2+1j*2*b*Mirror_in.z);
        E=np.exp(-1j*k*R-k*b)/R*(1+np.cos(theta))/2/k/w0*b;
        E=E*np.sqrt(8);
                
        cos_i=np.abs(Mirror_in.x*Mirror_n.x+Mirror_in.y*Mirror_n.y+Mirror_in.z*Mirror_n.z)/r;

        return E.real,E.imag,cos_i;
    
    else:
        Theta_max=Edge_angle;
        E_taper=Edge_taper;
        b=(20*np.log10((1+np.cos(Theta_max))/2)-E_taper)/(20*k*(1-np.cos(Theta_max))*np.log10(np.exp(1)));
        w0=np.sqrt(2/k*b)
        r,theta,phi=cart2spher(Mirror_in.x,Mirror_in.y,Mirror_in.z);
        co,cx,crho=CO(theta,phi);
        R=np.sqrt(r**2-b**2+1j*2*b*Mirror_in.z);
        F=np.exp(-1j*k*R-k*b)/R*(1+np.cos(theta))/2/k/w0*b;
        F=F*np.sqrt(8);
        if polarization.lower()=='x':
            E=scalarproduct(F,co);
            H=scalarproduct(F,cx);
        elif polarization.lower()=='y':
            H=scalarproduct(F,co);
            E=scalarproduct(F,cx);
        '''
        else:
            print('polarization input error');
        '''
        return E,H;
            
        
class Elliptical_GaussianBeam():
    def __init__(self,
                 Edge_taper,
                 Edge_angle,
                 k,
                 coor_angle,coor_displacement,
                 polarization='scalar'):
        '''
        param 1: 'Edge_taper' define ratio of maximum power and the edge power in the antenna;
        param 2: 'Edge_angle' is the angular size of the mirror seen from the feed coordinates;
        param 3: 'k' wave number;
        param 4: 'Mirror_in' the sampling points in the mirror illumanited by feed;
        param 5: 'fieldtype' chose the scalar mode or vector input field.
        '''
        self.Tx = Edge_taper[0]
        self.Ty = Edge_taper[1]
        self.Ax = Edge_angle[0]/180*np.pi
        self.Ay = Edge_angle[1]/180*np.pi
        
        self.coor_A = coor_angle
        self.coor_D = coor_displacement
        
        if polarization.lower()=='scalar':
            ax = (np.log10((1+np.cos(self.Ax))/2)-self.Tx/20)/((1-np.cos(self.Ax))*np.log10(np.exp(1)))
            by = (np.log10((1+np.cos(self.Ay))/2)-self.Ty/20)/((1-np.cos(self.Ay))*np.log10(np.exp(1)))

            def beam(Mirror,Mirror_n):
                Mirror=global2local(self.coor_A,self.coor_D,Mirror)
                Mirror_n=global2local(self.coor_A,[0,0,0],Mirror_n)
                r,theta,phi=cart2spher(Mirror.x,Mirror.y,Mirror.z)
                Amp = (1+np.cos(theta))*np.exp(-(1-np.cos(theta))*(ax*np.cos(phi)**2+by*np.sin(phi)**2))
                F = Amp *np.exp(-1j*k*r)/r
                cos_i=np.abs(Mirror.x*Mirror_n.x+Mirror.y*Mirror_n.y+Mirror.z*Mirror_n.z)/r
                #E = F/np.sqrt(np.sum(np.abs(F)**2))
                return F, cos_i
        else:
            ax = (np.log10((1+np.cos(self.Ax))/2)-self.Tx/20)/((1-np.cos(self.Ax))*np.log10(np.exp(1)))
            by = (np.log10((1+np.cos(self.Ay))/2)-self.Ty/20)/((1-np.cos(self.Ay))*np.log10(np.exp(1)))
            def beam(Mirror,Mirror_n):
                Mirror=global2local(self.coor_A,self.coor_D,Mirror)
                Mirror_n=global2local(self.coor_A,[0,0,0],Mirror_n)
                r,theta,phi=cart2spher(Mirror.x,Mirror.y,Mirror.z)
                Amp = (1+np.cos(theta))*np.exp(-(1-np.cos(theta))*(ax*np.cos(phi)**2+by*np.sin(phi)**2))
                F = Amp *np.exp(-1j*k*r)/r
                co,cx,crho=CO(theta,phi);
                if polarization.lower()=='x':
                    E=scalarproduct(F,co)
                    H=scalarproduct(F,cx)
                elif polarization.lower()=='y':
                    H=scalarproduct(F,co)
                    E=scalarproduct(F,cx)
                return E,H
        self.beam = beam
  
    
    

