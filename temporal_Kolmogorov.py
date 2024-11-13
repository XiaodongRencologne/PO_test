#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
import matplotlib.pyplot as plt;


# In[42]:


def phase_noise(dt,N,phase_err):
    '''
    step 1: produce time array
    '''
    t=np.linspace(0,(N-1)*dt,N);
    freq=np.fft.fftfreq(N,d=dt);
    
    '''
    step 2: produce random phase error
    '''
    dp=phase_err/180*np.pi #degree
    phase=np.random.normal(0,dp,N);
    F_phase=np.fft.fft(phase);
    P_phase=np.abs(F_phase)**2;
    
    '''
    step 3: modify the power spectrum
    q^(-11/3)
    '''
    Freq=freq;
    NN=np.where(Freq==0);
    Freq[NN]=Freq[NN[0]+1];
    Modify=np.abs(Freq)**(-8/3);
    Modify=Modify/Modify[NN]
    P=np.sqrt(P_phase*Modify)*np.exp(1j*np.angle(F_phase));

    '''
    step 4: get the modified random phase noise
    '''
    Phase=np.fft.ifft(P);
    Phase=Phase.real-Phase.real.mean()
    Phase=Phase*(dp/Phase.std());
    
    return freq,Phase;