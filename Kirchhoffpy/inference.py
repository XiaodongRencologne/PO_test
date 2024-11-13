#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as T;
import numpy as np;


# In[2]:


DEVICE0=T.device('cpu')

# define the class of complex;
class Complex:
    def __init__(self):
        self.real=T.tensor([]).to(DEVICE0);
        self.imag=T.tensor([]).to(DEVICE0);
        
    def np2Tensor(self,DEVICE):
        self.real=(self.real).to(DEVICE);
        self.imag=(self.imag).to(DEVICE);
        


# In[4]:


'''
1. convert all of the data type to GPU OR 
'''
def DATA2CUDA(*arguments,DEVICE):
    Argu=(());
    for data in arguments:
        if isinstance(data,T.Tensor):
            data=data.to(DEVICE);
            Argu=Argu+(data,);
        elif isinstance(data,np.ndarray) or isinstance(data,np.float64):
            data=T.tensor(data).to(DEVICE);
            Argu=Argu+(data,);
        else:
            data.np2Tensor(DEVICE);
            Argu=Argu+(data,);
    return Argu;


# In[5]:


'''
2. define the error model
'''

def error_ff(adjuster,List,p,q,m):
    
    """
    :para 1: 'adjuster'  changes for each adjusters
    :para 2: 'List'      center of list 
    :para 3: 'p q '      p q determine position of adjuster
    :para 4: ' m  '      mirror of model
    :para 5: 'size'      size=[sizex,sizey]
    """
    
    #m=copy.deepcopy(m0);
    ad=adjuster.clone();
    
    x=m.x.view(List.shape[0],-1).clone();
    y=m.y.view(List.shape[0],-1).clone();
    ad=ad.reshape(5,-1);
    s1=ad[0,...].view(-1,1).clone();
    s2=ad[1,...].view(-1,1).clone();
    s3=ad[2,...].view(-1,1).clone();
    s4=ad[3,...].view(-1,1).clone();
    s5=ad[4,...].view(-1,1).clone();
    
    a=s1.clone();
    b=(s2-s3-s4+s5)/4/p;
    c=(s2+s3-s4-s5)/4/q;
    d=((s2+s3+s4+s5)/4-s1)/(p**2+q**2);
    f=d.clone();
    e=(s2-s3+s4-s5)/4/p/q;
    
    x=x-List[...,0].view(-1,1)
    y=y-List[...,1].view(-1,1)
   
    dz=(a+ x*b + y*c + x**2*d + x*y*e +y**2*f);
    
    dz=dz.view(m.x.size()).clone();
    
    return dz;


# In[ ]:


'''
3.  DATA pre-processing corrected the beam pattern by the center point in the map;
'''
def correctphase(data): 
    N_angle=int(np.sqrt(data.size()[1]));
    data1=T.zeros(data.size(),dtype=T.float64).to(DEVICE0);
    for i in range(0,data.size()[0]-1,2):
        
        Phase0=T.atan2(data[i+1,...].view(N_angle,-1)[int(N_angle/2),int(N_angle/2)],data[i,...].view(N_angle,-1)[int(N_angle/2),int(N_angle/2)])
        Amp0=T.sqrt(data[i+1,...].view(N_angle,-1)[int(N_angle/2),int(N_angle/2)]**2+data[i,...].view(N_angle,-1)[int(N_angle/2),int(N_angle/2)]**2)
        
        c=1/Amp0*T.cos(-Phase0);
        s=1/Amp0*T.sin(-Phase0);
        
        data1[i,...]=data[i,...]*c-data[i+1,...]*s;
        data1[i+1,...]=data[i,...]*s+data[i+1,...]*c;
        
    return data1;


# In[ ]:


'''

4. function used for fitting;
'''
def fitting_func(M1,M3,# two perpared matrixes;
                 cosm2_i,cosm2_r,cosm1_i,cosm1_r,# reflection and incident angles;
                 Field_m2,adjuster,# input the field on m2
                 List_m2,List_m1,m2,m1,p_m2,q_m2,p_m1,q_m1,
                 k,Para_A,Para_p,aperture_xy):
    '''
    # parameters for large scale error in amplitude;
    Amp*(1+u*x,v*y+w*x*y+s*x^2+t*y^2);
    phi=phi0+a*x+b*y+c*(x**2)+d*(y**2); 
    '''
    amp=Para_A[0];u=Para_A[1];v=Para_A[2];w=Para_A[3];s=Para_A[4];t=Para_A[5];
    # phase terms caused by the large-scale error
    phi0=Para_p[0];a=Para_p[1];b=Para_p[2];c=Para_p[3];d=Para_p[4];
        
    # adjusters;        
    ad2=adjuster[0:5*List_m2.size()[0]].clone();    
    ad1=adjuster[5*List_m2.size()[0]:].clone();
   
    # distorted of panels;
    dz1=error_ff(ad1,List_m1,p_m1,q_m1,m1);   
    dz2=error_ff(ad2,List_m2,p_m2,q_m2,m2);
        
    Phase=T.atan2(Field_m2.imag,Field_m2.real);
    Phase=Phase+k*(dz2*(cosm2_i.view(1,-1))+dz2*cosm2_r.view(1,-1)); # corrected the phase on m2 based on dz of mirror
    Amp=T.sqrt(Field_m2.real**2+Field_m2.imag**2);
    
    Field=Complex();
    Field.real=Amp*T.cos(Phase);
    Field.imag=Amp*T.sin(Phase);
    # get field on m1
    Field_m1=Complex();
    Field_m1.real=(T.mm(M1.real,Field.real.view(-1,1))-T.mm(M1.imag,Field.imag.view(-1,1))).view(1,-1);
    Field_m1.imag=(T.mm(M1.imag,Field.real.view(-1,1))+T.mm(M1.real,Field.imag.view(-1,1))).view(1,-1);
    # aperture coordinates xy 
    
    x=aperture_xy[0,...].reshape(Field_m1.real.size());
    y=aperture_xy[1,...].reshape(Field_m1.real.size());
    
    Rm=amp*(1+u*x+v*y+w*x*y+s*x**2+t*y**2);   
    phi=phi0+a*x+b*y+c*(x**2)+d*(y**2);            
    
    
    Phase=T.atan2(Field_m1.imag,Field_m1.real)+phi;
    Phase=Phase+k*(dz1*(cosm1_i.view(1,-1))+dz1*cosm1_r.view(1,-1)); # corrected the phase on m1 based on dz of mirror
    Amp=T.sqrt(Field_m1.real**2+Field_m1.imag**2)*Rm;
    
    Field.real=Amp*T.cos(Phase);
    Field.imag=Amp*T.sin(Phase);
    
    Field_s=Complex();
    Field_s.real=(T.mm(M3.real,Field.real.view(-1,1))-T.mm(M3.imag,Field.imag.view(-1,1))).view(1,-1);
    Field_s.imag=(T.mm(M3.imag,Field.real.view(-1,1))+T.mm(M3.real,Field.imag.view(-1,1))).view(1,-1);
    
    
       
    data=T.cat((Field_s.real,Field_s.imag))
    
    return data;    

