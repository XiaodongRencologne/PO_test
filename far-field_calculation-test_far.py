#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch as T
import copy 
import time
import matplotlib.pyplot as plt

import PyccatPO;
from PyccatPO import field_calculation, field_calculation_far

import Kirchhoffpy
from Kirchhoffpy.Spheical_field import spheical_grid
from Kirchhoffpy.coordinate_operations import Coord



# In[6]:


# define the parameters input files
inputfile='CCAT_model'
sourcefile='beam'
defocus=[0,0,0]
'''
ad=np.genfromtxt('CCAT_model/fitting_error.txt');
ad_m2=ad[0:5*69];
ad_m1=ad[5*69:];
'''
ad_m2=np.zeros(5*69);
ad_m1=np.zeros((5,77));

#source_field=spheical_grid(-0.005,0.005,-0.005,0.005,Ns,Ns,distance=300*10**3)
source=Coord();
source = spheical_grid(-0.005,0.005,-0.005,0.005,1001,201,FIELD='far',Type='uv')
print(source.x)
#np.savetxt(sourcefile+'/beam.txt',  source.x)
#source0=np.genfromtxt(sourcefile+'/beam.txt');
#source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_fimag_E,Field_fimag_H,Field_m1_E,Field_m1_H,Field_s_E,Field_s_H=field_calculation_far(inputfile,
                                                                                            source,
                                                                                            defocus,
                                                                                            ad_m2,
                                                                                            ad_m1)
Ns=int(np.sqrt(source.x.size))
Sx=(Field_s_E.x.real+1j*Field_s_E.x.imag).reshape(Ns,Ns)
Sy=(Field_s_E.y.real+1j*Field_s_E.y.imag).reshape(Ns,Ns)
Sz=(Field_s_E.z.real+1j*Field_s_E.z.imag).reshape(Ns,Ns)

N_IF = int(np.sqrt(Field_fimag_H.x.real.size))
IF_x = (Field_fimag_H.x.real + 1j*Field_fimag_H.x.imag).reshape(N_IF,-1)
IF_y = (Field_fimag_H.y.real + 1j*Field_fimag_H.y.imag).reshape(N_IF,-1)
IF_z = (Field_fimag_H.z.real + 1j*Field_fimag_H.z.imag).reshape(N_IF,-1)


# In[9]:

outputfolder = 'output/296GHz/far_field/'

fig=plt.figure(figsize=(8,7));
plt.pcolor(20*np.log10(np.abs(Sx)));
plt.xlabel('far-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig(outputfolder + 'far_field_beam.png')

fig=plt.figure(figsize=(8,7));
plt.pcolor(20*np.log10(np.abs(Sy)));
plt.xlabel('far-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig(outputfolder + 'far_field_beam_y.png')

fig=plt.figure(figsize=(8,7));
plt.pcolor(20*np.log10(np.abs(Sz)));
plt.xlabel('far-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig(outputfolder + 'far_field_beam_z.png')



fig=plt.figure(figsize=(8,7));
plt.pcolor(20*np.log10(np.abs(IF_x)));
plt.xlabel('far-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig(outputfolder + 'IF_field_beam.png')

fig=plt.figure(figsize=(8,7));
plt.pcolor(20*np.log10(np.abs(IF_y)));
plt.xlabel('far-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig(outputfolder + 'IF_field_beam_y.png')

fig=plt.figure(figsize=(8,7));
plt.pcolor(20*np.log10(np.abs(IF_z)));
plt.xlabel('far-feild beam in amplitude (dB)',fontsize=18,color='darkred')
plt.colorbar();
plt.savefig(outputfolder + 'IF_field_beam_z.png')
# In[14]:


# saveing data;
data=np.concatenate((Sx.real,Sx.imag,Sy.real,Sy.imag,Sz.real,Sz.imag)).reshape(6,-1).T
# saveing data;
np.savetxt(outputfolder +'source_field.txt',data,delimiter=',');

# saveing data;
data=np.concatenate((IF_x.real,IF_x.imag,IF_y.real,IF_y.imag,IF_z.real,IF_z.imag)).reshape(6,-1).T
# saveing data;
np.savetxt(outputfolder +'source_field.txt',data,delimiter=',');






