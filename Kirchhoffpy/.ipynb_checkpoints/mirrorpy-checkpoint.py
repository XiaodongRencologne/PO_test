#!/usr/bin/env python
# coding: utf-8

# In[61]:



"""
Package used to build mirror model based on 'uniform sampling' and 'Gaussian quadrature';

"""

import numpy as np;
import torch as T;
import scipy
from .coordinate_operations import Coord;

from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;
# In[ ]:


'''
Define square panel
'''
'''
#1. define the surface function and the function also can calculate the normal vector z is negetive.
'''
def profile(coefficients,R):
    '''
    param 'R': is normalized factor;
    param 'coefficients': used to describe surface by 2-d polynomial surface. 
    '''
    def surface(x,y):
        z=np.polynomial.polynomial.polyval2d(x/R,y/R,coefficients);
        Mn=Coord();
        Mn.z=-np.ones(x.shape);
        a=np.arange(coefficients.shape[0]);
        c=coefficients*a.reshape(-1,1);
        Mn.x=np.polynomial.polynomial.polyval2d(x/R,y/R,c[1:,:])/R;    
        a=np.arange(coefficients.shape[1]);
        c=coefficients*a;
        Mn.y=np.polynomial.polynomial.polyval2d(x/R,y/R,c[:,1:])/R;
        N=np.sqrt(Mn.x**2+Mn.y**2+1);
        Mn.x=Mn.x/N;
        Mn.y=Mn.y/N;
        Mn.z=Mn.z/N;
        Mn.N=N;
        return z,Mn;    
    return surface;
'''
#2 define the panel reflector by using the surface function and panel location and size;
'''        
# define the panel     
def squarepanel(centerx,centery,sizex,sizey,Nx,Ny,surface,quadrature='uniform'):
    centerx=np.array(centerx);
    centery=np.array(centery);
    sizex=np.array(sizex);
    sizey=np.array(sizey);
    Nx=np.array(int(Nx));
    Ny=np.array(int(Ny))
        
    if quadrature.lower()=='uniform':
        x=np.linspace(-sizex/2+sizex/Nx/2,sizex/2-sizex/Nx/2,Nx);
        y=np.linspace(-sizey/2+sizey/Ny/2,sizey/2-sizey/Ny/2,Ny);
        xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2));
            
        x=xyarray[...,0];
        y=xyarray[...,1];
            
        M=Coord();
        M.x=(centerx.reshape(-1,1)+x).ravel();
        M.y=(centery.reshape(-1,1)+y).ravel();
        # surface can get the z value of the mirror and also can get the normal 
        M.z,Mn=surface(M.x,M.y);
        dA=sizex*sizey/Nx/Ny;
    elif quadrature.lower()=='gaussian':
        
        print(1);
    
        
    return M,Mn,dA;

'''
# 3 define the inaginary field size, we also can use the above function to do that.
'''
# define the imaginary flatten plane
def ImagPlane(Rangex,Rangey,Nx,Ny):
    fimag=Coord();
    dx=Rangex/(Nx-1);
    dy=Rangey/(Ny-1);
    dA=dx*dy;
    
    P=np.moveaxis(np.mgrid[-Rangex/2:Rangex/2:Nx*1j,-Rangey/2:Rangey/2:Ny*1j],0,-1);
    # fimag.x fimag.y fimag.z;
    fimag.x=P[...,1].ravel();fimag.y=P[...,0].ravel();
    fimag.z=np.zeros(fimag.x.shape);
    
    # normal vector
    fimagn=Coord();
    fimagn.x=np.zeros(fimag.x.shape)
    fimagn.y=np.zeros(fimag.x.shape)
    fimagn.z=np.ones(fimag.x.shape)
    fimagn.N=np.ones(fimag.x.shape);
    
    return fimag,fimagn,dA;    

'''
# 4 combine the panel function and surface function, and a special funtion for the two-mirror system;
'''
'''
def model_ccat(coefficient_m2,List_m2,M2_sizex,M2_sizey,M2_Nx,M2_Ny,   # m2
          coefficient_m1,List_m1,M1_sizex,M1_sizey,M1_Nx,M1_Ny,R, # m1
          Rangex,Rangey,fimag_Nx,fimag_Ny,# imaginary field
          S_init,p_m2,q_m2,p_m1,q_m1):              #fimag & initial position of adjusters;
    
    surface_m2=profile(coefficient_m2,R);# define the surface function of m2;
    surface_m1=profile(coefficient_m1,R);# define the surface function of m1;    
    m2,m2_n,m2_dA=squarepanel(List_m2[...,0],List_m2[...,1],M2_sizex,M2_sizey,M2_Nx,M2_Ny,surface_m2);
    m1,m1_n,m1_dA=squarepanel(List_m1[...,0],List_m1[...,1],M1_sizex,M1_sizey,M1_Nx,M1_Ny,surface_m1);
    fimag,fimag_n,fimag_dA=ImagPlane(Rangex,Rangey,fimag_Nx,fimag_Ny);
    
    # modified the panel based on the initial adjusters distribution;
    Panel_N_m2=int(List_m2.size/2)
    S_init=S_init.ravel();
    S_m2=S_init[0:Panel_N_m2*5];
    S_m1=S_init[Panel_N_m2*5:];
    
    m2_dz=deformation(S_m2,List_m2,p,q,m2);
    m1_dz=deformation(S_m1,List_m1,p,q,m1);
    
    m2.z=m2.z+m2_dz;
    m1.z=m1.z-m1_dz;
    
    return m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA;        
            
'''
'''
# 5 put regular error into the panel
'''
def deformation(adjuster,List,p,q,mirror):
    """
    :para 1: 'adjuster'  changes for each adjusters
    :para 2: 'List'      center of list 
    #:para 3: 'List_vector' normal vector in the center of the panel;
    :para 4: 'p q '      p q determine position of adjuster;
    :para 5: ' mirror  '      mirrors;
    """
    ad=T.tensor(adjuster);
    x=T.tensor(mirror.x).view(int(List.size/2),-1);
    y=T.tensor(mirror.y).view(int(List.size/2),-1);
    ad=ad.view(5,-1);
    s1=ad[0,...].view(-1,1);
    s2=ad[1,...].view(-1,1);
    s3=ad[2,...].view(-1,1);
    s4=ad[3,...].view(-1,1);
    s5=ad[4,...].view(-1,1);
    
    # error model use the value of each adjuster to model the deformation patterns;
    a=s1;
    b=(s2-s3-s4+s5)/4/p;
    c=(s2+s3-s4-s5)/4/q;
    d=((s2+s3+s4+s5)/4-s1)/(p**2+q**2);
    f=d;
    e=(s2-s3+s4-s5)/4/p/q;
    
    # get the
    x=x-T.tensor(List[...,0].reshape(-1,1));
    y=y-T.tensor(List[...,1].reshape(-1,1));
    
    dz=a+ x*b + y*c + (x**2)*d + x*y*e +(y**2)*f;
    
    dz=dz.data.numpy();
    dz=dz.reshape(mirror.z.shape)
    
    return dz;
            
'''
# 6 define the position of adjusters
'''   
def adjuster(List_m2,List_m1,p_m2,q_m2,p_m1,q_m1,R2,R1):
    # define the coordinates of adjusters in mirrors;
    #ad_m2_xy=np.zeros((2,int(List_m2.size/2)*5));
    #ad_m1_xy=np.zeros((2,int(List_m1.size/2)*5));
    p_m2=np.array(p_m2);p_m2=np.array(p_m2);R2=np.array(R2);
    p_m1=np.array(p_m1);q_m1=np.array(q_m1);R1=np.array(R1);
    List_m2=np.array(List_m2);List_m1=np.array(List_m1)
    ad1=List_m2.T;
    ad2=ad1.copy(); ad2[0,...]=ad2[0,...]+p_m2;ad2[1,...]=ad2[1,...]+q_m2;
    ad3=ad1.copy(); ad3[0,...]=ad3[0,...]-p_m2;ad3[1,...]=ad3[1,...]+q_m2;
    ad4=ad1.copy(); ad4[0,...]=ad4[0,...]-p_m2;ad4[1,...]=ad4[1,...]-q_m2;
    ad5=ad1.copy(); ad5[0,...]=ad5[0,...]+p_m2;ad5[1,...]=ad5[1,...]-q_m2;
    ad2_xy=np.append(ad1,ad2,axis=1);
    ad2_xy=np.append(ad2_xy,ad3,axis=1);
    ad2_xy=np.append(ad2_xy,ad4,axis=1);
    ad2_xy=np.append(ad2_xy,ad5,axis=1); 
    #ad2_xy=T.tensor(ad2_xy)#.to(DEVICE);
    x2=ad2_xy[0,...]/R2;
    y2=ad2_xy[1,...]/R2;
    
    ad1=List_m1.T
    ad2=ad1.copy(); ad2[0,...]=ad2[0,...]+p_m1;ad2[1,...]=ad2[1,...]+q_m1;
    ad3=ad1.copy(); ad3[0,...]=ad3[0,...]-p_m1;ad3[1,...]=ad3[1,...]+q_m1;
    ad4=ad1.copy(); ad4[0,...]=ad4[0,...]-p_m1;ad4[1,...]=ad4[1,...]-q_m1;
    ad5=ad1.copy(); ad5[0,...]=ad5[0,...]+p_m1;ad5[1,...]=ad5[1,...]-q_m1;
    ad1_xy=np.append(ad1,ad2,axis=1);
    ad1_xy=np.append(ad1_xy,ad3,axis=1);
    ad1_xy=np.append(ad1_xy,ad4,axis=1);
    ad1_xy=np.append(ad1_xy,ad5,axis=1);  
    #ad1_xy=T.tensor(ad1_xy)#.to(DEVICE);
    x1=ad1_xy[0,...]/R1;
    y1=ad1_xy[1,...]/R1;
    
    return x2,y2,x1,y1;        
        


# another function;        
def random_ad(N_ad2,N_ad1,x2,y2,x1,y1,rms=100):
    
    ad2=np.random.normal(0,rms,N_ad2)/1000;
    ad1=np.random.normal(0,rms,N_ad1)/1000;
    
    def dz(paras,x,y):
        z=paras[0]+paras[1]*x+paras[2]*y+paras[3]*x*y+paras[4]*x**2+paras[5]*y**2;
        return z;
    def ad_f(paras):
        z=dz(paras,x2,y2);
        r=((ad2-z)**2).sum();
        return r;
    
    para0=np.zeros(6);
    para=scipy.optimize.minimize(ad_f,para0,method='BFGS',tol=1e-6);
    ad2=ad2-dz(para.x,x2,y2);
    
    def ad_f(paras):
        z=dz(paras,x1,y1);
        r=((ad1-z)**2).sum();
        return r;
    para=scipy.optimize.minimize(ad_f,para0,method='BFGS',tol=1e-6);
    ad1=ad1-dz(para.x,x1,y1);
    
    return np.append(ad2,ad1);    
    

    

