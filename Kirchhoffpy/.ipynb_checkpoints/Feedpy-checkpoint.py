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
from .Vopy import vector,scalarproduct,sumvector, dotproduct,CO

c=299792458;
mu=4*np.pi*10**(-7);
epsilon=8.854187817*10**(-12);
Z0=np.sqrt(mu/epsilon,dtype = np.float64)

'''        
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
'''
def Gaussian2d(theta_x_2,theta_y_2,theta, phi):
    Amp = np.exp(-theta**2*np.cos(phi)**2/theta_x_2 - theta**2*np.sin(phi)**2/theta_y_2)
    return Amp

def Normal_factor(theta_x_2,theta_y_2):
    theta = np.linspace(0,np.pi,10001)
    dt = theta[1]-theta[0]
    phi = np.linspace(0,2*np.pi,10001)
    dp = phi[1]-phi[0]
    E = Gaussian2d(theta_x_2,theta_y_2,theta,phi)
    P = np.sum(np.abs(E)**2*np.sin(theta)*dt*dp)
    print(P)
    Nf = np.sqrt(1/P)
    print(Nf)
    print(np.sum(np.abs(Nf*E)**2*np.sin(theta)*dt*dp))
    return Nf
    

'''
Type 1: Gaussian beam;
'''

class GaussiBeam():
    def __init__(self,
                 Edge_taper,
                 Edge_angle,
                 k,
                 coord_sys,
                 #coor_angle,
                 #coor_displacement,
                 polarization='scalar'):
        
        self.T = Edge_taper
        self.A = Edge_angle/180*np.pi
        
        #self.coor_A = coor_angle
        #self.coor_D = coor_displacement
        self.coord_sys = coord_sys

        b = (np.log10((1+np.cos(self.A))/2)-self.T/20)/(k*(1-np.cos(self.A))*np.log10(np.exp(1)))
        w_2 = 2/k*(20*np.log10((1+np.cos(self.A))/2)-self.T)/(20*k*(1-np.cos(self.A))*np.log10(np.exp(1)))
        b = k*w_2/2
        w = np.sqrt(w_2)
        theta_2 = -20*self.A**2/self.T*np.log10(np.exp(1))
        
        if polarization.lower()=='scalar':
            def beam(Mirror,Mirror_n):
                Mirror=global2local(self.coor_A,self.coor_D,Mirror)
                Mirror_n=global2local(self.coor_A,[0,0,0],Mirror_n)
                r,theta,phi=cart2spher(Mirror.x,Mirror.y,Mirror.z)
                R=np.sqrt(r**2-b**2+1j*2*b*Mirror_in.z)
                E=np.exp(-1j*k*R-k*b)/R*(1+np.cos(theta))/2/k/w0*b
                E=E*np.sqrt(8)
                cos_i=np.abs(Mirror.x*Mirror_n.x+Mirror.y*Mirror_n.y+Mirror.z*Mirror_n.z)/r;
                return E.real,E.imag,cos_i;
        else: 
            B = 2*np.pi*np.exp(-2*b*k)/4/b**3/k**3
            B = B*(np.exp(4*b*k)*(8*b**2*k**2-4*b*k+1)-1)
            print(B)
            #Nf = np.sqrt(Z0/B)
            #Nf = 1/k/np.sqrt(B)
            Nf = np.sqrt(4*np.pi/k**2/B)
            def beam(Mirror,Mirror_n):
                #Mirror=global2local(self.coor_A,self.coor_D,Mirror)
                #Mirror_n=global2local(self.coor_A,[0,0,0],Mirror_n)
                r,theta,phi=cart2spher(Mirror.x,Mirror.y,Mirror.z)
                print('Gain of the Guassian feedhonr!')
                print(np.log10((Nf*((1+np.cos(theta)) * np.exp(k*b*np.cos(theta))).max())**2/Z0*4*np.pi)*10,'dB')
                F = (1+np.cos(theta)) * np.exp(k*b*np.cos(theta)) * np.exp(-1j*k*r)/r
                F = Nf*F
                #print((1/Z0*np.abs(F)**2*Mirror.w*Mirror_n.N).sum())
                print((k**2*np.abs(F)**2*Mirror.w*Mirror_n.N).sum())
                E = vector()
                H = vector()
                co,cx,crho=CO(theta,phi)
                if polarization.lower()=='x':
                    E=scalarproduct(F,co)
                    #H=scalarproduct(F/Z0,cx)
                    H=scalarproduct(F,cx)
                elif polarization.lower()=='y':
                    #H=scalarproduct(F/Z0,co)
                    E=scalarproduct(F,cx)
                    H=scalarproduct(F,co)
                return E, H
        self.source = beam

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
            H=scalarproduct(F/Z0,cx);
        elif polarization.lower()=='y':
            H=scalarproduct(F/Z0,co);
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

        bx = (np.log10((1+np.cos(self.Ax))/2)-self.Tx/20)/(k*(1-np.cos(self.Ax))*np.log10(np.exp(1)))
        by = (np.log10((1+np.cos(self.Ay))/2)-self.Ty/20)/(k*(1-np.cos(self.Ay))*np.log10(np.exp(1)))
        print(bx,by)
        wx_2 = 2/k*(20*np.log10((1+np.cos(self.Ax))/2)-self.Tx)/(20*k*(1-np.cos(self.Ax))*np.log10(np.exp(1)))
        wy_2 = 2/k*(20*np.log10((1+np.cos(self.Ay))/2)-self.Ty)/(20*k*(1-np.cos(self.Ay))*np.log10(np.exp(1)))
        bx = k*wx_2/2
        by = k*wy_2/2
        wx = np.sqrt(wx_2)
        wy = np.sqrt(wy_2)
        #print(bx,by)
        #print(wx,wy)
        print(2/k/wx*180/np.pi,2/k/wy*180/np.pi)
        theta_x_2 = -20*self.Ax**2/self.Tx*np.log10(np.exp(1))
        theta_y_2 = -20*self.Ay**2/self.Ty*np.log10(np.exp(1))
        print(np.sqrt(theta_x_2)*180/np.pi,np.sqrt(theta_y_2)*180/np.pi)
        Nf = Normal_factor(theta_x_2 ,theta_y_2)
        if polarization.lower()=='scalar':
            def beam(Mirror,Mirror_n):
                Mirror=global2local(self.coor_A,self.coor_D,Mirror)
                Mirror_n=global2local(self.coor_A,[0,0,0],Mirror_n)
                r,theta,phi=cart2spher(Mirror.x,Mirror.y,Mirror.z)
                
                w_x_2 = wx_2*(1+(Mirror.z/bx)**2)
                w_y_2 = wy_2*(1+(Mirror.z/by)**2)
                Amp_x = -1j/bx*np.exp(k*bx)*wx/np.sqrt(w_x_2)*np.exp(-Mirror.x**2/w_x_2)
                Amp_y = -1j/by*np.exp(k*by)*wy/np.sqrt(w_y_2)*np.exp(-Mirror.y**2/w_y_2)
                R_x = Mirror.z*(1+(bx/Mirror.z)**2)
                #R_y = Mirror.z*(1+(by/Mirror.z)**2)
                Amp = Amp_x*Amp_y
                F = Amp * np.exp(-1j*(k*(Mirror.x**2+Mirror.y**2)/2/R_x \
                                      + k*Mirror.z - np.arctan(Mirror.z/bx)/2 - np.arctan(Mirror.z/by)/2))
                cos_i=np.abs(Mirror.x*Mirror_n.x+Mirror.y*Mirror_n.y+Mirror.z*Mirror_n.z)/r
                #E = F/np.sqrt(np.sum(np.abs(F)**2))
                return F, cos_i
        else: 
            def beam(Mirror,Mirror_n):
                Mirror=global2local(self.coor_A,self.coor_D,Mirror)
                Mirror_n=global2local(self.coor_A,[0,0,0],Mirror_n)
                r,theta,phi=cart2spher(Mirror.x,Mirror.y,Mirror.z)
                F = Nf*Gaussian2d(theta_x_2 ,theta_y_2, theta, phi) * np.exp(-1j*k*r)/r
                E = vector()
                H = vector()
                co,cx,crho=CO(theta,phi);
                #co,cx,crho=CO(theta,phi);
                if polarization.lower()=='x':
                    E=scalarproduct(F,co);
                    H=scalarproduct(F/Z0,cx);
                    E_co = F#dotproduct(E,co)
                    E_cx = 0#dotproduct(E,cx)
                    E_r  = 0#dotproduct(E,crho)
                elif polarization.lower()=='y':
                    H=scalarproduct(F/Z0,co)
                    E=scalarproduct(F,cx)
                    E_co = 0#dotproduct(E,co)
                    E_cx = F#dotproduct(E,cx)
                    E_r  = 0#dotproduct(E,crho)
                return E, H , E_co , E_cx
        self.beam = beam
