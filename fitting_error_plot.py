#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np;
import matplotlib.pyplot as plt;

from Pyccat import model_ccat,read_input;
from Kirchhoffpy.mirrorpy import deformation,adjuster;

'''
read input data file
'''
inputfile='CCAT_model'
coefficient_m2,coefficient_m1,List_m2,List_m1,M2_size,M1_size,R2,R1,p_m2,q_m2,p_m1,q_m1,M2_N,M1_N,fimag_N,fimag_size,distance,edge_taper,Angle_taper,k=read_input(inputfile);

M2_size=M2_size+1.2;
M1_size=M1_size+1.2;
M2_N=[13,13]
M1_N=[13,13]
'''
build model
'''
ad_m2=np.zeros(5*List_m2.shape[0]);
ad_m1=np.zeros(5*List_m1.shape[0]);
m2,m2_n,m2_dA,m1,m1_n,m1_dA,fimag,fimag_n,fimag_dA=model_ccat(coefficient_m2,List_m2,M2_size[0],M2_size[1],M2_N[0],M2_N[1],R2,
                                                                  coefficient_m1,List_m1,M1_size[0],M1_size[1],M1_N[0],M2_N[1],R1,
                                                                  fimag_size[0],fimag_size[1],fimag_N[0],fimag_N[1],
                                                                  ad_m2,ad_m1,p_m2,q_m2,p_m1,q_m1);

dx2,dy2,dx1,dy1=adjuster(List_m2,List_m1,p_m2,q_m2,p_m1,q_m1,R2,R1)


'''
define function to reshape the mirror to mesh grid points;
'''
def reshape_model(sizex,sizey,m,z,dy=1,Num=11):
    x0=np.linspace(-4.5*sizex+sizex/Num/2,4.5*sizex-sizex/Num/2,Num*9)
    y0=np.linspace(-4.5*sizey+sizey/Num/2,4.5*sizey-sizey/Num/2,Num*9)
    #m2
    y0=y0+dy;
    #m1
    #y0=y0;
    x,y=np.meshgrid(x0,y0)
    dz=np.zeros(x.shape)
    for i in range(9*Num):
        for n in range(9*Num):        
            a=np.where((m.x>(x[i,n]-0.001))&(m.x<(x[i,n]+0.001)) &(m.y>(y[i,n]-0.001))&(m.y<(y[i,n]+0.001)))        
            if a[0].size:
                dz[i,n]=z[a];
            else:
                dz[i,n]=np.nan;
    return x,y,dz


# In[18]:


'''
define a color map plot function;
'''
def colormap(x1,y1,z1,x2,y2,z2,Vmax=None,Vmin=None,savename='',suptitle=''):
    cmap = plt.get_cmap('hot');
    font = {'family': 'serif','color':'darkred','weight':'normal','size':16};
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(14,6));
    ax1=ax[0];
    ax2=ax[1];
    p1=ax1.pcolor(x2,y2,z2,cmap=cmap,vmin=Vmin,vmax=Vmax);
    ax1.axis('scaled');
    ax1.set_xlabel('Secondary mirror',fontdict=font);
    clb=fig.colorbar(p1, ax=ax1,shrink=0.95,fraction=.05);
    clb.set_label('um',labelpad=-40,y=1.05,rotation=0)
    p1=ax2.pcolor(x1,y1,z1,cmap=cmap,vmin=Vmin,vmax=Vmax);
    ax2.axis('scaled');
    ax2.set_xlabel('Primary mirror',fontdict=font);
    clb=fig.colorbar(p1, ax=ax2,shrink=0.95,fraction=.05);
    clb.set_label('um',labelpad=-40,y=1.05,rotation=0);
    fig.suptitle(suptitle,fontsize=15,color='k',verticalalignment='top')#'baseline')
    plt.savefig('output/picture/'+savename+'.png')
    plt.show();
    
'''
define a 5 adjuster plots
'''
def ad_plots(err,name,scale=10):
    fig,ax=plt.subplots(nrows=2,ncols=3,figsize=(12,12));
    font = {'family': 'serif',
            'color':'darkred',
            'weight':'normal',
            'size':16};
    ax1=ax[0,0];
    ax2=ax[0,1]
    ax3=ax[0,2]
    ax4=ax[1,0]
    ax5=ax[1,1]
    ax6=ax[1,2];
    ax6.set_visible(False);
    
    err2=err[0:List_m2.shape[0]*5].reshape(5,-1);
    err1=err[List_m2.shape[0]*5:].reshape(5,-1);
    
    ax1.plot(err2[0,...],'b*-',label='M2 adjuster 0');
    ax1.plot(err1[0,...],'ro-',label='M1 adjuster 0');
    ax1.set_ylim([-scale,scale])
    ax1.set_ylabel('Fitting error/$\mu m$')
    ax1.legend();
    
    ax2.plot(err2[1,...],'b*-',label='M2 adjuster 1');
    ax2.plot(err1[1,...],'ro-',label='M1 adjuster 1');
    ax2.set_ylim([-scale,scale])
    ax2.legend();
    
    ax3.plot(err2[2,...],'b*-',label='M2 adjuster 2');
    ax3.plot(err1[2,...],'ro-',label='M1 adjuster 2');
    ax3.set_ylim([-scale,scale])
    ax3.legend();
    
    ax4.plot(err2[3,...],'b*-',label='M2 adjuster 3');
    ax4.plot(err1[3,...],'ro-',label='M1 adjuster 3');
    ax4.set_ylim([-scale,scale])
    ax4.set_ylabel('Fitting error/$\mu m$')
    ax4.legend();
    
    ax5.plot(err2[4,...],'b*-',label='M2 adjuster 4');
    ax5.plot(err1[4,...],'ro-',label='M1 adjuster 4');
    ax5.set_ylim([-scale,scale])
    ax5.legend();
    
    plt.savefig('output/picture/fitting_adjuster_errorplots'+name+'.png')
    plt.show()
    
    


# In[19]:


'''
Define a function to plot error maps
1. input panel error;
2. fitting panel error;
3. fitting accuracy;
4. final error plots;
5. error calculation;
'''
def error_plots(file_input,file_fitting,name,inputrms=100,outputrms=10,scale=10):
    # 1. get input panel error (reference):
    ad0=np.genfromtxt(file_input)
    ad_m2=ad0[0:5*List_m2.shape[0]];
    ad_m1=ad0[5*List_m2.shape[0]:];
    ''' reshape the error matrixes shape'''
    M20=deformation(ad_m2.ravel(),List_m2,p_m2,q_m2,m2);
    M10=deformation(ad_m1.ravel(),List_m1,p_m1,q_m1,m1);
    x2,y2,dz2_0=reshape_model(M2_size[0],M2_size[1],m2,M20,dy=-1,Num=int(M2_N[0]));
    x1,y1,dz1_0=reshape_model(M1_size[0],M1_size[1],m1,M10,dy=35,Num=int(M1_N[0]));
    #2. panel error from holography fitting
    ad1=np.genfromtxt(file_fitting)[0:5*int((List_m2.size+List_m1.size)/2)];
    ad_m2=ad1[0:5*List_m2.shape[0]]
    ad_m1=ad1[5*List_m2.shape[0]:5*(List_m2.shape[0]+List_m1.shape[0])];
    ''' reshape the error matrixes shape'''    
    M20=deformation(ad_m2.ravel(),List_m2,p_m2,q_m2,m2);
    M10=deformation(ad_m1.ravel(),List_m1,p_m1,q_m1,m1);
    x2,y2,dz2_1=reshape_model(M2_size[0],M2_size[1],m2,M20,dy=-1,Num=int(M2_N[0]));
    x1,y1,dz1_1=reshape_model(M1_size[0],M1_size[1],m1,M10,dy=35,Num=int(M1_N[0]));
    
    # 3. calculate the error;    
    err2=(dz2_0-dz2_1)*1000;
    err1=(dz1_0-dz1_1)*1000;
    rms2=np.sqrt(np.nanmean(err2**2));
    rms1=np.sqrt(np.nanmean(err1**2));
    
    '''
    1. input panel error map;
    '''
    colormap(x1,y1,dz1_0*1000,x2,y2,dz2_0*1000,Vmax=inputrms,Vmin=-inputrms,savename='Input_panel_error'+name,suptitle='Panel deformation map');
    '''
    2. fitting results of the panel error;
    '''
    colormap(x1,y1,dz1_1*1000,x2,y2,dz2_1*1000,Vmax=inputrms,Vmin=-inputrms,savename='fitting_panel_error'+name,suptitle='Fitting results');
    
    '''
    3. fitting error
    '''
    colormap(x1,y1,err1,x2,y2,err2,Vmax=outputrms,Vmin=-outputrms,savename='fitting_error'+name,suptitle='Error distribution');
    
    '''
    4. fitting adjuster error cut plots
    '''
    ad_plots((ad1-ad0).ravel()*1000,name,scale=scale);
    
    print(' M2 error:',rms2,'um\n','M1 error',rms1,'um');
    
    
    


# In[ ]:




