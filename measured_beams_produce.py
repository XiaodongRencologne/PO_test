#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np;
import torch as T;
import copy ;
import time;
import matplotlib.pyplot as plt;

import Pyccat;
from Pyccat import field_calculation;
from Pyccat import Make_fitfuc

import Kirchhoffpy;
from Kirchhoffpy.Spheical_field import spheical_grid;
from Kirchhoffpy.coordinate_operations import Coord;



# In[6]:


# define the parameters input files
inputfile='CCAT_model';
sourcefile='beam'
ad_m2=np.zeros(5*69);
ad_m1=np.zeros((5,77));
S_init=np.genfromtxt('CCAT_model/panel_error_296GHz.txt')
#S_init=np.genfromtxt('panel_error2.txt')
ad_m2=S_init[0:5*69];
ad_m1=S_init[5*69:];

#source_field=spheical_grid(-0.005,0.005,-0.005,0.005,Ns,Ns,distance=300*10**3)
source=Coord();
defocus=[400,400,600];
print('1')
source0=np.genfromtxt(sourcefile+'/400_400_600.txt');
Ns=int(np.sqrt(source0[...,0].size))
source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_s,Field_fimag,Field_m1,Field_m2=field_calculation(inputfile,source,defocus,ad_m2,ad_m1);
S1=(Field_s.real+1j*Field_s.imag).reshape(Ns,Ns);
defocus=[400,-400,600];
print('2')
source0=np.genfromtxt(sourcefile+'/400_-400_600.txt');
source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_s,Field_fimag,Field_m1,Field_m2=field_calculation(inputfile,source,defocus,ad_m2,ad_m1);
S2=(Field_s.real+1j*Field_s.imag).reshape(Ns,Ns);
defocus=[-400,400,600];
print('3')
source0=np.genfromtxt(sourcefile+'/-400_400_600.txt');
source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_s,Field_fimag,Field_m1,Field_m2=field_calculation(inputfile,source,defocus,ad_m2,ad_m1);
S3=(Field_s.real+1j*Field_s.imag).reshape(Ns,Ns);
defocus=[-400,-400,600];
print('4')
source0=np.genfromtxt(sourcefile+'/-400_-400_600.txt');
source.x=source0[...,0];source.y=source0[...,1];source.z=source0[...,2];
Field_s,Field_fimag,Field_m1,Field_m2=field_calculation(inputfile,source,defocus,ad_m2,ad_m1);
S4=(Field_s.real+1j*Field_s.imag).reshape(Ns,Ns);

# saveing data;
np.savetxt('input_beams/measured_beam_maps_51/pospos.txt',np.append(S1.real,S1.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('input_beams/measured_beam_maps_51/posneg.txt',np.append(S2.real,S2.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('input_beams/measured_beam_maps_51/negpos.txt',np.append(S3.real,S3.imag).reshape(2,-1).T,delimiter=',');
np.savetxt('input_beams/measured_beam_maps_51/negneg[.txt',np.append(S4.real,S4.imag).reshape(2,-1).T,delimiter=',');
S=np.concatenate((S1.real,S1.imag,S2.real,S2.imag,S3.real,S3.imag,S4.real,S4.imag)).reshape(8,-1)
#np.savetxt('input_beams/measured_beam_maps/perfect_4p_beams.txt',S,delimiter=',');
np.savetxt('input_beams/measured_beam_maps_51/err_4p_beams.txt',S,delimiter=',');

# In[ ]:





# In[ ]:




