#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 21:36:11 2023

@author: rezad.d.esfahani


"""

import matplotlib.pyplot as plt
import numpy as np
import RWDC as RW





freq = np.arange(0.001, 1, 0.005)


H = np.array([1, 2, 3, 4, 10]);
beta = 1.8* np.array([0.7, 1.5, 1.5,  1.3, 1.6, 1.7]);
alpha = 3* np.asarray([1, 1.4, 1.4,  1.2, 1.5, 1.7]);
rho = 1.75* np.array([1, 1, 1.5,  1.5, 1.5, 1.6]);


H = np.array([1, 1, 1, 1, 1, 1, 1,1, 2, 2, 3, 4, 10]) ;      
beta = 1.* np.array([0.7, 0.8, 0.9 ,1, 1.2, 1.3, 1.4 , 1.5, 1.5,  1.6,1.7,1.8, 1.9, 2])
alpha = 3* np.asarray([1, 1.1,  1.2, 1.3, 1.4, 1.4, 1.4,1.5, 1.6, 1.7, 1.8, 1.8, 1.9, 2]);
rho = 1.75* np.array([1, 1,1,1 ,1,1,1, 1, 1,1,  1.5,  1.5, 1.5, 1.6])


import time 


t = time.time()

vr = RW.Rayleghwave(freq, H, rho, alpha, beta, mode= 4);

elapsed = time.time() - t

print(elapsed)


plt.plot(freq, vr)

#%%


def a1(x):
    return RW.Rayleghwave(freq, H, rho, alpha, x, mode= 1)



#%%


cc = sp.optimize.minimize(secular, np.array(k2), args = (om, thk, rho, vp, vs) )




#%%

    vp = alpha
    vs = beta
    
    thk = H
    
    # Find Vmin and Vmax
    vpmin = np.min(vp)
    vsmin = np.min(vs)
    
    vpmax = np.max(vp)
    vsmax = np.max(vs)
    
    #Shear wave velocity in homogenous material
    crmin = HHS(vpmin, vsmin)
    crmax = HHS(vpmax, vsmax)
    
    crmin = 0.98 * crmin
    crmax = 1 * vsmax


    MAXROOT = mode
    # Number of wavenumber samples between Kmin and Kmax
    NUMINC = 2000
    
    TOL = 1e-10
    
    nfreq = len(freq)
    nlayer = len(vs)
    
    cr = np.zeros([nfreq, MAXROOT])
    
    # For each frequency find the best K
    for i in range(nfreq):
        
        
        if nlayer == 1:
            cr[i, 0] = HHS(vp, vs)
            
        else:
            
            numroot = 0
            
            om = 2 * np.pi * freq[i]
            
            
            kmax = om/ crmin
            kmin = om/crmax
            
            dk = (kmax - kmin) / NUMINC
            
            k1 = kmax 
            
            t = time.time()

            f1 = secular(k1, om, thk, rho, vp, vs)


            elapsed = time.time() - t
            
            print(elapsed)


            k2 = kmax - dk
            
            f2 = secular(k2, om, thk, rho, vp, vs)
            
            
            kold = 1 * kmax
            
            
            for j in range(2, NUMINC, 1):
            
                k3 = kmax - j * dk
                f3 = secular(k3, om, thk, rho, vp, vs)#func(k3)
                
                if (f2 < f1) & (f2 < f3):
                    
                    
                    kminfound, fval, ierr, numfunc = fminbound(secular, k3, k1,args=(om, thk, rho, vp, vs), xtol=1e-10, maxfun = 1e5, full_output=True)

                    if (fval<TOL) & (abs((kminfound-kold)/kold) > TOL):
                        
                        cr[i, numroot] = om/kminfound
                        kold = kminfound
                        numroot = numroot +1
                        
                    if numroot == MAXROOT-1:
                        break
