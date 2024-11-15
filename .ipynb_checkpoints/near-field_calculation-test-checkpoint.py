#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
import torch as T;
import copy ;
import time;
import matplotlib.pyplot as plt;

import PyccatPO;
from PyccatPO import field_calculation;

import Kirchhoffpy;
from Kirchhoffpy.Spheical_field import spheical_grid;
from Kirchhoffpy.coordinate_operations import Coord;



# In[6]:


# define the parameters input files
inputfile='CCAT_model';
sourcefile='beam'
defocus=[0,0,600];
ad=np.genfromtxt('CCAT_model/fitting_error.txt');
ad_m2=ad[0:5*69];
ad_m1=ad[5*69:];
ad_m2=np.zeros(5*69);
ad_m1=np.zeros((5,77));

#source_field=spheical_grid(-0.005,0.005,-0.005,0.005,Ns,Ns,distance=300*10**3)
source=Coord();
source0=np.genfromtxt(sourcefile+'/on-axis.txt',delimiter=',');
source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_fimag_E,Field_fimag_H,Field_m1_E,Field_m1_H,Field_s_E,Field_s_H=field_calculation(inputfile,source,defocus,ad_m2,ad_m1);
Ns=int(np.sqrt(source.x.size));
Sx=(Field_s_E.x.real+1j*Field_s_E.x.imag).reshape(Ns,Ns);
Sy=(Field_s_E.y.real+1j*Field_s_E.y.imag).reshape(Ns,Ns);
Sz=(Field_s_E.z.real+1j*Field_s_E.z.imag).reshape(Ns,Ns);

# In[9]:


fig=plt.figure(figsize=(8,7));
plt.pcolor(source.x.reshape(Ns,Ns),source.y.reshape(Ns,Ns),20*np.log10(np.abs(Sx)));
plt.xlabel('near-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig('output/near_field3/near_field_beam.png')

fig=plt.figure(figsize=(8,7));
plt.pcolor(source.x.reshape(Ns,Ns),source.y.reshape(Ns,Ns),20*np.log10(np.abs(Sy)));
plt.xlabel('near-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig('output/near_field3/near_field_beam_y.png')

fig=plt.figure(figsize=(8,7));
plt.pcolor(source.x.reshape(Ns,Ns),source.y.reshape(Ns,Ns),20*np.log10(np.abs(Sz)));
plt.xlabel('near-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig('output/near_field3/near_field_beam_z.png')

# In[14]:

Source=np.concatenate((Sx.real,Sx.imag,Sy.real,Sy.imag,Sz.real,Sz.imag)).reshape(6,-1).T
# saveing data;

np.savetxt('output/near_field3/source_field.txt',Source,delimiter=',');
#np.savetxt('output/near_field3/imaginary_field.txt',np.append(Field_fimag.real,Field_fimag.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/near_field3/m1_field.txt',np.append(Field_m1.real,Field_m1.imag).reshape(2,-1).T,delimiter=',');
#np.savetxt('output/near_field3/m2_field.txt',np.append(Field_m2.real,Field_m2.imag).reshape(2,-1).T,delimiter=',');






