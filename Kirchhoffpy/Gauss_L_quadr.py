
# coding: utf-8

# In[66]:


"""
Build gauss_legendre quadrature sampling points and weight
"""
import numpy as np;

# 1D gaissoam quadrature integration. 
def Gauss_L_quadrs1d(start, stop, N_part,N):
    """
    :parameter a：the start integration points;
    :parameter b: the end integration points;
    :parameter N_part: the uniform parts that you want to seprate;
    :parameter N: the sampling number for each part.
    """
    step=(stop-start)/N_part;
    line=np.arange(start,stop+step,step);
    
    x_0,w_0=np.polynomial.legendre.leggauss(N);#sampling points and weight for (-1,1);
    w_0=w_0*step/2;
    x_0=x_0*step/2;
    
    x=np.array([]);
    w=np.array([]);
    for n in range(N_part):
        x=np.append(x,x_0+(line[n]+line[n+1])/2);
        w=np.append(w,w_0);
    
    return x,w

def Gauss_L_quadrs2d(x0,x1,Nx_part,Nx,y0,y1,Ny_part,Ny):
    x,wx=Gauss_L_quadrs1d(x0,x1,Nx_part,Nx)
    y,wy=Gauss_L_quadrs1d(y0,y1,Ny_part,Ny)

    xyarray=np.reshape(np.moveaxis(np.meshgrid(x,y),0,-1),(-1,2))
    xarray=np.transpose(xyarray[:,0])
    yarray=np.transpose(xyarray[:,1])
    warray=np.reshape(np.moveaxis(np.meshgrid(wx,wy),0,-1),(-1,2))  
    w=warray[:,0]*warray[:,1]
    return xarray,yarray,w



def Guass_L_quadrs_Circ(a,r_phi,
                        Nr_part,Nr,
                        phi0,phi1,
                        N_phi,
                        Phi_type='uniform'):
    '''circular elliptical aperture, 

    #######
    r=rho*[r_0(phi)-a]+a
    phi=phi
    Sum=Sum(F(x,y)|N|)*|r_0(phi)-a|*r*dr*d_phi
    #########

    the integration in radiual direction uses Gaussian L quadrature.
    trapz rule is used in the angular direction integration. 
    **r0=a
    **r1 is a function of phi
    **Nr_part: Rho direction is devided into a few uniform section.
    ** Nr is sampling points in each section
    *** phi0 phi1 is the angular integration range.
    *** N_phi 
    '''
    # sampling radiual direction
    rho,w0=Gauss_L_quadrs1d(0,1,Nr_part,Nr)
    if Phi_type=='uniform':
        phi=np.linspace(phi0,phi1,N_phi)
        phi,rho=np.meshgrid(phi,rho)
        
        w=np.ones((Nr_part*Nr,N_phi))
        w[:,0]=1/2;w[:,-1]=1/2
        w=w*(phi1-phi0)/(N_phi-1)
        w0=np.repeat(w0,N_phi).reshape(-1,N_phi)
        w=(w*w0).ravel()

        phi=phi.ravel()
        rho=rho.ravel()*(r_phi(phi)-a)+a
        w=w*(r_phi(phi)-a)*rho

    elif Phi_type=='less':
        N_phi_p=np.int_(np.round((max(N_phi,10)-10)*np.sqrt(rho))+10)
        rho=np.repeat(rho.ravel(),N_phi_p)
        w0=np.repeat(w0.ravel(),N_phi_p)
        phi=np.array([])
        w=np.array([])

        for item in N_phi_p:
            phi=np.append(phi,np.linspace(phi0,phi1,item))
            w_phi=np.ones(item); w_phi[0]=1/2; w_phi[-1]=1/2
            w_phi=w_phi*(phi1-phi0)/(item-1)
            w=np.append(w,w_phi)
        rho=rho*(r_phi(phi)-a)+a
        w=w*w0*(r_phi(phi)-a)*rho

    return rho*np.cos(phi),rho*np.sin(phi),w
    