#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
from .coordinate_operations import Coord;
from .coordinate_operations import Transform_local2global as local2global;
from .coordinate_operations import Transform_global2local as global2local;


# In[ ]:


def spheical_grid(umin,umax,vmin,vmax,Nu,Nv,FIELD='near',Type='uv',filename='',distance=300*10**3):
    FIELD=FIELD.lower();
    cut=Coord();
    if filename=='':
        grid=np.moveaxis(np.meshgrid(np.linspace(umin,umax,Nu),np.linspace(vmin,vmax,Nv)),0,-1);
        cut.x=grid[...,0].ravel();
        cut.y=grid[...,-1].ravel();
    else:
        grid=read_angular_grid(filename);
        cut.x=grid[...,0].ravel()/180*np.pi;
        cut.y=grid[...,1].ravel()/180*np.pi;
        
    Grid_type={'uv':     lambda x,y: (x,y,np.sqrt(1-(x**2+y**2))),
               'EloverAz':lambda x,y: (-np.sin(x)*np.cos(y),np.sin(y),np.cos(x)*np.cos(y))
              }
    #cutw=model();
    cut.x,cut.y,cut.z=Grid_type[Type](cut.x,cut.y);
    
    if FIELD=='far':
        pass;
    elif FIELD=='near':
        cut.x=distance*cut.x;
        cut.y=distance*cut.y;
        cut.z=distance*cut.z;
        #cut=local2global(angle,displacement,cut);
    else:
        print("don't support FIELD=",FIELD);
    
    
    return cut;


# In[ ]:




