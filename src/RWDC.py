#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  20 15:40:57 2022

@author: rezad.d.esfahani

"""

from scipy.optimize import fminbound, root

import numpy as np
import scipy as sp




import time 


def Rayleghwave(freq,thk,rho,vp,vs, mode = 3):
    
    # Find Vmin and Vmax
    vpmin = np.min(vp)
    vsmin = np.min(vs)
    
    vpmax = np.max(vp)
    vsmax = np.max(vs)
    
    #Shear wave velocity in homogenous material
    crmin = HHS(vpmin, vsmin)
    crmax = HHS(vpmax, vsmax)
    
    crmin = 0.9 * crmin
    crmax = 1 * vsmax
    
    
    cr = Rayleighmodel(freq,thk,rho,vp,vs, crmin, crmax, mode = mode)
    
    return cr




def Rayleighmodel(freq,thk,rho,vp,vs, crmin, crmax, mode = 3):
    
    """
    Estimate phase velocity of Rayleigh wave 

    Parameters
    ----------
    freq : TYPE
        DESCRIPTION.
    thk : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
    vp : TYPE
        DESCRIPTION.
    vs : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    cr : TYPE
        DESCRIPTION.

    """
    

    MAXROOT = mode
    # Number of wavenumber samples between Kmin and Kmax
    NUMINC = 1000
    
    TOL = 0.001
    
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
            


            f1 = secular(k1, om, thk, rho, vp, vs)




            k2 = kmax - dk
            
            f2 = secular(k2, om, thk, rho, vp, vs)
            
            
            kold = 1.1 * kmax
            
            
            for j in range(1, NUMINC, 1):
            
                k3 = kmax - j * dk
                f3 = secular(k3, om, thk, rho, vp, vs)#func(k3)
    
                if (f2 < f1) & (f2 < f3):
                    
                    kminfound, fval, ierr, numfunc = fminbound(secular, k3, k1,args=(om, thk, rho, vp, vs), xtol=1e-15, full_output=True)

                    if (fval<TOL) & (abs((kminfound-kold)/kold) > TOL):
                        
                        cr[i, numroot] = om/kminfound
                        kold = kminfound
                        numroot = numroot +1
                        
                    if numroot == MAXROOT:
                        break
                        
                k1 = k2
                k2 = k3
                f1 = f2
                f2 = f3
                
    return cr


def HHS(vp , vs):
    """
    Rayleigh wave velocity in Homogenous media

    Parameters
    ----------
    vp : TYPE
        DESCRIPTION.
    vs : TYPE
        DESCRIPTION.

    Returns
    -------
    cvr : TYPE
        DESCRIPTION.

    """
        
    
    nu = 0.5 * (vp**2 - 2 * vs**2) / (vp**2 - vs**2)
    
    c = 8* (3 - 2 * (vs**2 / vp**2))
    
    d = 16 * (((vs**2) / (vp**2))-1)
    
    
    p = [1, -8, c, d]
        
    x = np.roots(p)
    
    ph = vs * np.sqrt(x)
    
    
    crest = vs * (0.862 + 1.14 * nu) / (1 + nu)
    
    
    # Determine which roots is correct (Achenbach, 1973) Need to read 
    index = np.where(abs(ph - crest) == np.min(abs(ph - crest)))[0]
    
    if len(index)>1:
        index = index[0]
    
    cvr = abs(ph[index])
    

    return cvr 


def secular(k, om, thk, rho, vp, vs):
    """
    Calculate the secular function for specific frequency and wavenumer

    Parameters
    ----------
    k : TYPE
        DESCRIPTION.
    om : TYPE
        DESCRIPTION.
    thk : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
    vp : TYPE
        DESCRIPTION.
    vs : TYPE
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    
    epsilon = 1e-3
    while any(abs(om/k-vs)<epsilon) | any(abs(om/k-vp)<epsilon):
        k = k * (1+epsilon)
        
    
    
    e11,e12,e21,e22,du,mu,nus,nup = PSV(thk, rho, vp, vs, om, k)
    
    td,tu,rd,ru = modelrt(e11,e12,e21,e22,du)

    Td,Rd = genrt(td, tu, rd, ru)
    
    d = abs(sp.linalg.det(e21[:, :, 0] + e22[:, :, 0] @ du[:, : , 0] @ Rd[:, :, 0]) / (nus[0] * nup[0] * mu[0]**2))
    
    return d
    #print(1)
    

def PSV(thk, rho, vp, vs, om, k):
    """
    Calculate the downgoing and upgoing waves

    Parameters
    ----------
    thk : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
    vp : TYPE
        DESCRIPTION.
    vs : TYPE
        DESCRIPTION.
    om : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    e11 : TYPE
        DESCRIPTION.
    e12 : TYPE
        DESCRIPTION.
    e21 : TYPE
        DESCRIPTION.
    e22 : TYPE
        DESCRIPTION.
    du : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    nus : TYPE
        DESCRIPTION.
    nup : TYPE
        DESCRIPTION.

    """
    
    vs2  = vs**2 
    vp2 = vp**2
    
    mu = rho * (vs2)
    
    
    e11 = np.zeros([2, 2, len(vs)], dtype = np.complex128)
    e12 = np.zeros([2, 2, len(vs)], dtype = np.complex128)
    e21 = np.zeros([2, 2, len(vs)], dtype = np.complex128)
    e22 = np.zeros([2, 2, len(vs)], dtype = np.complex128)    
    du= np.zeros([2, 2, len(thk)], dtype = np.complex128)
    
    if om == 0:
        kappa = (1 + vs2/vp2) / (1- vs2/vp2)
        kmu = k * mu 
        
        
        e11[0, 0, :] = 1
        e11[0, 1, :] = e11[0, 0, :]
        e12[0, 0, :] = e11[0, 0, :]
        e12[0, 1, :] = e11[0, 0, :]
        
        
        e11[1, 0, :] = - (kappa - 1) 
        e11[1, 1, :] = e11[0, 0, :]
        e12[1, 0, :] = - e11[1, 0, :]
        e12[1, 1, :] = - e11[0, 0, :]
        
        e21[0, 0, :] = (kappa - 3) * kmu
        e21[0, 1, :] = -2 *kmu
        e22[0, 0, :] = -e21[0, 0, :]
        e22[0, 1, :] = -e21[0, 1, :]
        
        e21[1, 0, :] = (kappa - 1) * kmu
        e21[1, 1, :] = -2 * kmu
        e22[1, 0, :] = e21[1, 0, :]
        e22[1, 1, :] = e21[1, 1, :]
        
        du[0, 0, :] = np.exp(-k * thk)
        du[1, 1, :] = np.exp(-k * thk)
        du[1, 0, :] = -k *thk * np.exp(-k * thk)
        
    else:
        
        k2 = k**2
        om2 = om**2
        
        ks2 = om2/vs2
        nus = np.sqrt(np.complex128(k2 - ks2))
        index = np.squeeze(np.where(np.imag(-1j* nus) >0))
        nus[index] = -nus[index]
        gammas = nus / k
        
        
        kp2 = om2/vp2
        nup = np.sqrt(np.complex128( k2 - kp2))
        index = np.squeeze(np.where(np.imag(-1j* nup) >0))
        nup[index] = -nup[index]
        gammap = nup/k
        
        chi = 2 * k - ks2 / k
        
        
        e11[0, 0, :] = -1
        e11[0, 1, :] = gammas
        e12[0, 0, :] = e11[0, 0, :]
        e12[0, 1, :] = gammas
        
        e11[1, 0, :] = -gammap
        e11[1, 1, :] = -e11[0, 0, :]
        e12[1, 0, :] = gammap
        e12[1, 1, :] = e11[0, 0, :]
        
        e21[0, 0, :] = 2 * mu * nup
        e21[0, 1, :] = -mu * chi
        e22[0, 0, :] = -e21[0, 0, :]
        e22[0, 1, :] = -e21[0, 1, :]
        
        e21[1, 0, :] = -e21[0, 1, :]
        e21[1, 1, :] = -2* mu* nus
        e22[1, 0, :] = -e21[0, 1, :]
        e22[1, 1, :] = e21[1, 1, :]
        
        du[0, 0, :] = np.exp(-nup[0:len(thk)] * thk)
        du[1, 1, :] = np.exp(-nus[0:len(thk)] * thk)
        

        return e11,e12,e21,e22,du,mu,nus,nup
        

def modelrt(e11,e12,e21,e22,du):
    """
    
    Calculate the R/T coeff

    Parameters
    ----------
    e11 : TYPE
        DESCRIPTION.
    e12 : TYPE
        DESCRIPTION.
    e21 : TYPE
        DESCRIPTION.
    e22 : TYPE
        DESCRIPTION.
    du : TYPE
        DESCRIPTION.

    Returns
    -------
    td : TYPE
        DESCRIPTION.
    tu : TYPE
        DESCRIPTION.
    rd : TYPE
        DESCRIPTION.
    ru : TYPE
        DESCRIPTION.

    """
    
    [m, n, N]  = du.shape
    
    X = np.zeros([4, 4, N], dtype = np.complex128)
    
    for i in range(N-1):
        
        
        A = np.vstack(( np.hstack((e11[:, :, i+1], -e12[:, :, i])) , np.hstack((e21[:, :, i+1], -e22[:, :, i])) ))
        B = np.vstack(( np.hstack((e11[:, :, i], -e12[:, :, i+1])) , np.hstack((e21[:, :, i], -e22[:, :, i+1])) )) 
        L = np.vstack(( np.hstack((du[:, :, i], np.zeros([2, 2]))) , np.hstack((np.zeros([2, 2]), du[:, :, i+1])) ))
        
        X[:, :, i]  = sp.linalg.solve(A , B @ L)
        
    A = np.vstack(( np.hstack((e11[:, :, N], -e12[:, :, N-1])) , np.hstack((e21[:, :, N], -e22[:, :, N-1])) ))
    B = np.vstack(( e11[:, :, N-1] @ du[:, :, N-1] , e21[:, :, N-1] @ du[:, :, N-1] )) 
    
    X[:, 0:2, N-1] = sp.linalg.solve(A , B)
    
    td = X[0:2, 0:2, :]
    ru = X[0:2, 2:4, :]
    rd = X[2:4, 0:2, :]
    tu = X[2:4, 2:4, :]
    
    return td, tu, rd, ru
  

def genrt(td, tu, rd, ru):
    """
    
    Calculate the Generalized R/T coeff
    
    Parameters
    ----------
    td : TYPE
        DESCRIPTION.
    tu : TYPE
        DESCRIPTION.
    rd : TYPE
        DESCRIPTION.
    ru : TYPE
        DESCRIPTION.

    Returns
    -------
    Td : TYPE
        DESCRIPTION.
    Rd : TYPE
        DESCRIPTION.

    """
    
    [m, n, N] = td.shape
    
    Td =np.zeros([2, 2, N], np.complex128)
    Rd = np.zeros([2, 2, N], np.complex128)
    
    Td[:, :, N-1] = td[:,:, N-1]
    Rd[:, :, N-1] = rd[:, :, N-1]
    
    
    for i in range(N-2, -1, -1):
        
        Td[:, : , i] = sp.linalg.inv(np.eye(2) - ru[: , :, i] @ Rd[: , : , i+1]) @ td[:, : , i]
        Rd[:, : , i] = rd[:, :, i] + tu[:, :, i] @ Rd[:, :, i+1] @ Td[:, :, i]
    
    return Td,Rd

