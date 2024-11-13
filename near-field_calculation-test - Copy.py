#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
import torch as T;
import copy ;
import time;
import matplotlib.pyplot as plt;

import Pyccat2;
from Pyccat2 import field_calculation;
from Pyccat2 import Make_fitfuc

import Kirchhoffpy;
from Kirchhoffpy.Spheical_field import spheical_grid;
from Kirchhoffpy.coordinate_operations import Coord;



# In[6]:


# define the parameters input files
inputfile='CCAT_model';
sourcefile='beam'
defocus=[0,0,705];
ad=np.genfromtxt('CCAT_model/fitting_error.txt');
ad_m2=ad[0:5*69];
ad_m1=ad[5*69:];
#ad_m2=np.zeros(5*69);
#ad_m1=np.zeros((5,77));

#source_field=spheical_grid(-0.005,0.005,-0.005,0.005,Ns,Ns,distance=300*10**3)
source=Coord();
source0=np.genfromtxt(sourcefile+'/on-axis.txt',delimiter=',');
source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_s,Field_fimag,Field_m1,Field_m2=field_calculation(inputfile,source,defocus,ad_m2,ad_m1);
Ns=int(np.sqrt(source.x.size));
S=(Field_s.real+1j*Field_s.imag).reshape(Ns,Ns);


# In[9]:


fig=plt.figure(figsize=(8,7));
plt.pcolor(source.x.reshape(Ns,Ns),source.y.reshape(Ns,Ns),20*np.log10(np.abs(S)));
plt.xlabel('near-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig('output/near_field3/near_field_beam.png')


# In[14]:


# saveing data;
np.savetxt('output/near_field3/source_field.txt',np.append(S.real,S.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('output/near_field3/imaginary_field.txt',np.append(Field_fimag.real,Field_fimag.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('output/near_field3/m1_field.txt',np.append(Field_m1.real,Field_m1.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('output/near_field3/m2_field.txt',np.append(Field_m2.real,Field_m2.imag).reshape(2,-1).T,delimiter=',');


# In[ ]:




