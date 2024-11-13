#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#!/usr/bin/env python

# In[1]:


import numpy as np;
import matplotlib.pyplot as plt;


# In[22]:



'''1. square pattern'''
def squarePattern(u0,v0,urange,vrange,Nu,Nv,file='',distance='far',Type='on-axis'):
    if Type=='on-axis':
        grid=np.moveaxis(np.meshgrid(np.linspace(-urange/2,urange/2,Nu),np.linspace(-vrange/2,vrange/2,Nv)),0,-1);
        x=grid[...,0].ravel();
        y=grid[...,-1].ravel();
        z=np.sqrt(1-x**2-y**2);
        del(grid);
        if distance=='far':
            pass
        else:
            x=distance*x;
            y=distance*y;
            z=distance*z;
        grid=np.concatenate((x,y,z)).reshape(3,-1).T;
        np.savetxt(file+'/on-axis.txt',grid,delimiter=',');
        return 0;
    else:
        grid=np.moveaxis(np.meshgrid(np.linspace(-urange/2,urange/2,Nu),np.linspace(-vrange/2,vrange/2,Nv)),0,-1);
        x=grid[...,0].ravel();
        y=grid[...,-1].ravel();
        del(grid);
        ''' produce 4 beams'''
        # pos pos
        xpp=x+u0;
        ypp=y-v0;
        zpp=np.sqrt(1-xpp**2-ypp**2);
        # pos neg;
        xpn=x+u0;
        ypn=y+v0;
        zpn=np.sqrt(1-xpn**2-ypn**2);
        # neg pos
        xnp=x-u0
        ynp=y-v0
        znp=np.sqrt(1-xnp**2-ynp**2);
        # neg neg
        xnn=x-u0
        ynn=y+v0
        znn=np.sqrt(1-xnn**2-ynn**2);
        if distance=='far':
            grid=np.concatenate((xpp,ypp,zpp)).reshape(3,-1).T;
            np.savetxt(file+'/pos_pos_far.txt',grid);
            grid=np.concatenate((xpn,ypn,zpn)).reshape(3,-1).T;
            np.savetxt(file+'/pos_neg_far.txt',grid);
            grid=np.concatenate((xnp,ynp,znp)).reshape(3,-1).T;
            np.savetxt(file+'/neg_pos_far.txt',grid);
            grid=np.concatenate((xnn,ynn,znn)).reshape(3,-1).T;
            np.savetxt(file+'/neg_neg_far.txt',grid);
        else:
            grid=(np.concatenate((xpp,ypp,zpp)).reshape(3,-1).T)*distance;
            np.savetxt(file+'/pos_pos_near.txt',grid);
            grid=np.concatenate((xpn,ypn,zpn)).reshape(3,-1).T*distance;
            np.savetxt(file+'/pos_neg_near.txt',grid);
            grid=np.concatenate((xnp,ynp,znp)).reshape(3,-1).T*distance;
            np.savetxt(file+'/neg_pos_near.txt',grid);
            grid=np.concatenate((xnn,ynn,znn)).reshape(3,-1).T*distance;
            np.savetxt(file+'/neg_neg_near.txt',grid);
        return 1

  


'''2. plane field'''
def plane(sizex,sizey,Nx,Ny,distance,file=''):
    grid=np.moveaxis(np.meshgrid(np.linspace(-sizex/2,sizex/2,Nx),np.linspace(-sizey/2,sizey/2,Ny)),0,-1);
    x=grid[...,0].ravel();
    y=grid[...,-1].ravel();
    z=np.ones(x.size)*distance;
    grid=np.concatenate((x,y,z)).reshape(3,-1).T
    np.savetxt(file+'plane'+str(distance)+'mm.txt',grid);
    return 1;
    


# In[ ]:




