# %%
import numpy as np
from .Gauss_L_quadr import Gauss_L_quadrs2d,Gauss_L_quadrs1d,Guass_L_quadrs_Circ
import matplotlib.pyplot as plt

# %%
class Elliptical_rim():
    def __init__(self,Center,a,b,r_inner=0):
        self.cx=Center[0]
        self.cy=Center[1]
        self.a=np.abs(a)
        self.b=np.abs(b)
        self.e=np.sqrt(self.a**2-self.b**2)/self.a
        self.r_inner=r_inner
    def radial_profile(self,phi):
        R=self.b*np.sqrt(1/(1-self.e**2*np.cos(phi)**2))
        return R


    def sampling(self,
                 Nx,Ny,
                 quadrature='uniform',
                 Nr_part=1,phi0=0,phi1=2*np.pi,
                 Phi_type='uniform'):
        '''two sampling method for uniform integration and Guassian integration.
           Circular Gaussian intergration in r axis and angular direction using si
        '''
        if quadrature.lower()=='uniform':
            x,dx=np.linspace(-self.a/2+self.a/Nx/2,self.a/2-self.a/Nx/2,int(Nx),retstep=True)
            y,dy=np.linspace(-self.b/2+self.b/Ny/2,self.b/2-self.b/Ny/2,int(Ny),retstep=True)
            xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2))
            x=xyarray[:,0]
            y=xyarray[:,1]
            del(xyarray)
            NN=np.where(((x/(self.a/2))**2+(y/(self.b/2))**2)>1)
            x=np.delete(x,NN)
            y=np.delete(y,NN)
            x=x+self.cx
            y=y+self.cy
            dA=dx*dy
            w=dA
        elif quadrature.lower()=='gaussian':
            '''Nx=Nr, Ny=N_phi'''
            x,y,w=Guass_L_quadrs_Circ(self.r_inner,self.radial_profile,Nr_part,Nx,phi0,phi1,Ny,Phi_type=Phi_type)
            x=x+self.cx
            y=y+self.cy

        return x,y,w



class Rect_rim():
    def __init__(self, Center, a, b):
        '''
           center of rectangular rim
           a: x size
           b: y size
        '''
        self.cx=Center[0]
        self.cy=Center[1]
        self.sizex=np.abs(a)
        self.sizey=np.abs(b)

    def sampling(self,Nx,Ny,quadrature='uniform',Nx_part=1,Ny_part=1):
        '''two sampling method for uniform integration and Guassian integration'''
        if quadrature.lower()=='uniform':
            x,dx=np.linspace(-self.sizex/2+self.sizex/Nx/2,self.sizex/2-self.sizex/Nx/2,int(Nx),retstep=True)
            y,dy=np.linspace(-self.sizey/2+self.sizey/Ny/2,self.sizey/2-self.sizey/Ny/2,int(Ny),retstep=True)
            
            xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2))
            x=xyarray[:,0]+self.cx
            y=xyarray[:,1]+self.cy
            w=dx*dy
            return x,y,w 
        elif quadrature=='gaussian':
            x0=-self.sizex/2+self.cx
            x1=self.sizex/2+self.cx
            y0=-self.sizey/2+self.cy
            y1=self.sizey/2+self.cy
            x,y,w=Gauss_L_quadrs2d(x0,x1,Nx_part,Nx,y0,y1,Ny_part,Ny)
            return x,y,w    
        else:
            print('please input correct quadrature method.')
            return False



class Table_rect_rim():
    def __init__(self, c_list, a_list, b_list):
        '''
           c_list: center of each rectangular panels
           a_list: list of the xsize of panels
           b_list: list of the ysize of panels
        '''
        self.cx=c_list[0][:]
        self.cy=c_list[1][:]
        self.sizex=a_list
        self.sizey=b_list
    
    def sampling(self,Nx,Ny,quadrature='uniform'):
        '''
            Two sampling method for uniform integration and Guassian integration
            Nx and Ny are the sampling points list for all panels.
        '''
        X=np.array([],dtype=np.float64)
        Y=np.array([],dtype=np.float64)
        W=np.array([],dtype=np.float64)

        Num_panel=len(self.cx)
        if isinstance(Nx,int):
            Nx=[Nx]*Num_panel
            Ny=[Ny]*Num_panel
        if quadrature.lower()=='uniform':
            for n in range(Num_panel):
                x,dx=np.linspace(-self.sizex[n]/2+self.sizex[n]/Nx[n]/2,self.sizex[n]/2-self.sizex[n]/Nx[n]/2,int(Nx[n]),retstep=True)
                y,dy=np.linspace(-self.sizey[n]/2+self.sizey[n]/Ny[n]/2,self.sizey[n]/2-self.sizey[n]/Ny[n]/2,int(Ny[n]),retstep=True)
                xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2))
                X=np.append(X,xyarray[:,0]+self.cx[n])
                Y=np.append(Y,xyarray[:,1]+self.cy[n])
                W=np.append(W,np.array(dx*dy).repeat(Nx[n]*Ny[n]))
            return X,Y,W 
        elif quadrature=='gaussian':
            for n in range(Num_panel):
                x0=-self.sizex[n]/2+self.cx[n]
                x1=self.sizex[n]/2+self.cx[n]
                y0=-self.sizey[n]/2+self.cy[n]
                y1=self.sizey[n]/2+self.cy[n]
                print(x0,x1)
                print(y0,y1)
                x,y,w=Gauss_L_quadrs2d(x0,x1,1,Nx[n],y0,y1,1,Ny[n])
                X=np.append(X,x)
                Y=np.append(Y,y)
                W=np.append(W,w)
            return X,Y,W   
        else:
            print('please input correct quadrature method.')
            return False