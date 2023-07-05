#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:17:50 2023

@author: rezad.d.esfahani
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import (fft, ifft)
from scipy import interpolate



def ricker(f, dt):
    """
    Rciker function 
    
    """
    
    nw = 2./dt
    
    nw = 2* np.floor(nw/2)
    
    nc = np.floor(nw/2)
    
    k = np.arange(start = 1, stop= nw, step=1)
    
    alpha = (nc - k + 1)  * f* dt* np.pi
    
    beta = alpha **2
    
    w = (1- 2 * beta) * np.exp(-beta)
    
    tw = - (nc + 1 - np.arange(start = 1, stop= nw, step=1)) * dt
    
    return tw, w

def Pv2Gv(Pv, f):
    
    f = f.reshape(-1,)
    Pv = Pv.reshape(-1,)
    
    m = len(f)
    
    sp = 1/ Pv
    dsp = np.zeros_like(Pv)

    for i in range(m-1):
        dsp[i] = np.diff(sp[i:i+2]) / np.diff(f[i:i+2])
    
    sga=dsp * f+ sp
    
    Gv = 1 / sga[0:m-1]

    return Gv, f[0 : m-1]





def synthetic(f = 5.5, dt = 0.004, v0 = 250, dv = 180, sigma  = 20, decay = 50000, x  = np.arange(0, 1000, 8)):
    """
    
    Synthetic generation of surface wave based on dispersion curve based on
    
    
    """
    

    tw, sr = ricker(f,dt);
    fsr = fft(sr, n = 2500, axis = 0);
    fnq = 1/dt;
    nsr = len(fsr);
    
    
    f = np.linspace(0, fnq/2, nsr//2 )
    sf  = fsr[0: nsr//2]
    
    
    
    v = v0 + dv * np.exp(-f**2 / sigma**2);
    
    k= f/v;
    
    
    
    gv , f2 = Pv2Gv(v, f)
    
    
    beta = f/ decay
    
    h2 = np.zeros([nsr//2, len(x)], dtype = np.complex128)
        
    
    
    for i in range(nsr//2):
        
        h2[i,:] = sf[i] *  np.exp(-2 * 1j * np.pi * k[i] * x ) * np.exp(-(beta[i]) * (x)) 
    
    
    fn = np.real(ifft(h2, n = 1*nsr, axis = 0))
    
    fn = fn[len(tw)//2-1:-1]
    
    t = np.arange(0, len(fn[:,1]), 1) * dt
    
    
    plt.figure(figsize = (10, 3))
    plt.subplot(141)
    plt.plot(f, v)
    plt.xlabel('Frequency')
    plt.ylabel('Phase velocity')
    
    plt.subplot(142)
    plt.plot(f2, gv)
    plt.xlabel('Frequency')
    plt.ylabel('Group velocity')

    plt.subplot(143)
    plt.plot(f, beta)
    plt.xlabel('Frequency')
    plt.ylabel('Attenuation')

    plt.subplot(144)
    plt.plot(f, k)
    plt.xlabel('Frequency')
    plt.ylabel('Wavenumber')



    return t, fn, f, sf, v


def Slow2Vel(M, q):
    ynew = abs(np.zeros_like(M))
    
    xnew = np.linspace(1/q[-1], 1/q[0], len(q))

    for i in range(len(ynew[1,:])):
        
        
        x = 1/q
        y = M[:,i]
        f = interpolate.interp1d(x, y) 
        
        
        ynew[:,i] = f(xnew)
        
    return ynew
