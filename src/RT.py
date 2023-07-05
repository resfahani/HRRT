

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse.linalg import (lsqr, cg)

from scipy.fft import (fft, ifft)

from scipy import interpolate



def Inverse_Radon_Transform(m, h, dt, qmin, qmax):
    """
    Linear Inverse Radon Trasform (from Tau-p domain to the data domain)

    Parameters
    ----------
    m : Matrix (nt,nq) 
        Radon Panel.
        
    h : offset vector (nx) 
        nx.
    dt : sampling rate
    
    qmin : minimum velocity

    qmax : Maximum velocity
    
    
    Returns
    -------
    d : data domain
    
    """    
    
    nt, nq = np.shape(m)
    
    nx = len(h)
    
    
    q = np.linspace(qmin, qmax, nq)    
    
    
    nfft = 1 * next_power_of_2(nt)
    
    M = fft(m, n = nfft, axis = 0)
    
    
    ilow = 0
    ihigh = nfft//2

    D = np.zeros([ihigh, nx], dtype=np.complex128)
    f = 2 * np.pi /nfft/dt

    op = np.exp(1j * f * h[:,np.newaxis ] @ q[np.newaxis,:])

    op = np.conj(op)
    
    L = np.ones([nx, nq])
    
    I = np.eye(nx)

    for ifreq in range(ilow, ihigh):
        
        p = M[ifreq,:][:,np.newaxis]
        y = (L @ p)

        D[ifreq,:] = np.squeeze(y)

        L = L  * op


    d = 2 * ifft(D ,n = 1 * nfft ,axis = 0)
    d = np.real(d)
    d =d[:nt ,:]
    
    return d


def Radon_Transform(d, h, dt, qmin, qmax, nq, mode , mu = 1, gamma =1,  maxiter = 10):
    """
    Radon Trasform (from data domain to Tau-p domain)

    Parameters
    ----------
    d : Data domain (nt,nx)

    h : offset vector (nh)
        DESCRIPTION.
    dt : sampling rate (dt)
        DESCRIPTION.
    qmin : minimum velocity
        DESCRIPTION.
    qmax : maximum veloity
        DESCRIPTION.
    nq : velocity vector length
        DESCRIPTION.
    mode : adj :"Adjont "
            LS : "Least square lgorithm"
            IRLS: "Iterative reweighted least square"
            
        DESCRIPTION.
    mu: Regularization parameter
        DESCRIPTION
    maxiter : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    m : Radon domain (tau-nq)
        DESCRIPTION.

    """

    nt, nx = np.shape(d)
    q = np.linspace(qmin, qmax, nq)    

    nfft = 1 * next_power_of_2(nt)
    D = fft(d, n = nfft, axis = 0)
    
    ilow = 0
    ihigh = nfft//2

    M = np.zeros([nfft, nq], dtype=np.complex128)
    
    f = 2 * np.pi /nfft/dt

    op = np.exp(-1j * f * h[:,np.newaxis ] @ q[np.newaxis,:])
    
    
    L = np.ones([nx, nq])
    
    I = np.eye(nq) 
    
    for ifreq in range(ilow, ihigh):
        
        p = D[ifreq,:][:,np.newaxis]
        
        if   mode =='adj':

            y = (L.conj().T) @ p
            
        elif mode =='LS':
            rhs = (L.conj().T) @ p
            y = np.linalg.inv(L.conj().T @ L + I*mu) @ rhs
            
        elif mode == "IRLS":
            y = IRLS(L, p, gamma= mu, maxiter = maxiter)
            
            
        elif mode == "ADMM":
            
            y = ADMM(L, p, gamma=gamma, mu=mu, maxiter = maxiter)

        
        M[ifreq,:] = np.squeeze(y)
        
        L = L  * op

    m = 2 * ifft(M ,n = 1 * nfft ,axis = 0)
    m = np.real(m)
    m =m[:nt ,:]
    
    return m
    

def DCRT(d, h, dt, qmin, qmax, nq, mode , mu = 1, gamma =1,  maxiter = 10):
    """
    Radon Trasform (from data domain to Tau-p domain)

    Parameters
    ----------
    d : Data domain (nt,nx)

    h : offset vector (nh)
        DESCRIPTION.
    dt : sampling rate (dt)
        DESCRIPTION.
    qmin : minimum velocity
        DESCRIPTION.
    qmax : maximum veloity
        DESCRIPTION.
    nq : velocity vector length
        DESCRIPTION.
    mode : adj :"Adjont "
            LS : "Least square lgorithm"
            IRLS: "Iterative reweighted least square"
            
        DESCRIPTION.
    mu: Regularization parameter
        DESCRIPTION
    maxiter : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    m : Radon domain (tau-nq)
        DESCRIPTION.

    """
    
    d = d/ sp.linalg.norm(d)
    
    nt, nx = np.shape(d)
    q = np.linspace(qmin, qmax, nq)    

    nfft = 1 * next_power_of_2(nt)
    D = fft(d, n = nfft, axis = 0)
    
    ilow = 0
    ihigh = nfft//2

    M = np.zeros([nfft, nq], dtype=np.complex128)
    
    f = 2 * np.pi /nfft/dt

    op = np.exp(-1j * f * h[:,np.newaxis ] @ q[np.newaxis,:])
    
    
    L = np.ones([nx, nq])
    
    I = np.eye(nq) 
    
    for ifreq in range(ilow, ihigh):
        
        p = D[ifreq,:][:,np.newaxis]
        
        if   mode =='adj':

            y = (L.conj().T) @ p
            
        elif mode =='LS':
            rhs = (L.conj().T) @ p
            y = np.linalg.inv(L.conj().T @ L + I*mu) @ rhs
            
        elif mode == "IRLS":
            y = IRLS(L, p, gamma= mu, maxiter = maxiter)
            
            
        elif mode == "ADMM":
            
            y = ADMM(L, p, gamma=gamma, mu=mu, maxiter = maxiter)
            
        
        y = abs(np.squeeze(y))
        
        M[ifreq,:] = y / np.max(y)

        
        L = L  * op
        
        
    M , v = Slow2Vel(abs(M.T), q)
    
    return M, v

def Slow2Vel(M, q):
    ynew = abs(np.zeros_like(M))
    
    xnew = np.linspace(1/q[-1], 1/q[0], len(q))

    for i in range(len(ynew[1,:])):
        
        
        x = 1/q
        y = M[:,i]
        f = interpolate.interp1d(x, y) 
        
        
        ynew[:,i] = f(xnew)
        
    return ynew, xnew


def PhVelEst(m, dt):
    nt, nq = np.shape(m)
    nfft = 2 * next_power_of_2(nt)
    M = fft(m, n = nfft, axis = 0)
    ihigh = nfft//2
    f = np.arange(nfft//2)/nfft /dt
    
    M = np.abs(M[:ihigh,:].T)
    
    for i in range(ihigh):
        M[:,i]= M[:,i]/ np.max(M[:,i])
    
    return M, f



def next_power_of_2(n):
    """
    Return next power of 2 greater than or equal to n
    """
    return 2**(n-1).bit_length()




def IRLS(A, b, gamma, maxiter = 10):
    """
    
    Solve Ax = b
    
    based on 
    
    "Iterative Re-weighted least square algorithm Based on Accurate interpolation
    with high-resolution time-variant Radon transforms"
    
    and 
    
    "Regularization of geophysical ill-posed problems by iteratively re-weighted
    and refined least squares"

    Parameters
    ----------
    A : left-side matrix
        DESCRIPTION.
    b : righthand side
        DESCRIPTION.
    gamma : Regularization
        DESCRIPTION.
    maxiter : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    x : Result!
        DESCRIPTION.

    """
    n1, n2 = A.shape
    
    I = np.eye(n2)

    W = np.eye(n1)
    
    
    for _ in range(maxiter):
        
        rhs = gamma * A.conj().T.dot(W).dot(b)
        
        x = np.dot(np.linalg.inv(gamma *A.conj().T.dot(W).dot(A) + I ), rhs ) 
                
        e = A@x-b
        
        W =  np.diag(1/np.squeeze(np.sqrt(abs(e)**2 + 1e-5 )))**2
        I =  np.diag(1/np.squeeze(np.sqrt(abs(x)**2  + 1e-5 * np.max(abs(x)) )))**2
        
    return x


def ADMM(A, b, gamma, mu, maxiter = 10):
    
    # The alternating direction method of multipliers (ADMM)for the slant stacking (linear radon transform)
    #
    scale = sp.linalg.norm(b)
    
    b = b/ scale
    
    n1, n2 = A.shape
    
    I = np.eye(n2)

    LHS = np.linalg.inv( I + gamma *A.conj().T.dot(A))
    
    y0 = b.copy()
    
    lamnda1 = np.zeros([n2,1])
    
    lamda0 = np.zeros_like(b)
    
    y1 = np.zeros([n2,1])

    for _ in range(maxiter):
        
        x = np.dot(LHS, (y1 + lamnda1) + gamma * A.conj().T.dot(lamda0+y0))
        
        u = x - lamnda1
        
        mu3 = len(np.where(u.reshape(-1,)>np.median(abs(u)))[0])/len(u.reshape(-1,))
        mu2 = mu* mu3
        
        Ux = 1-( mu2 * np.max(np.abs(u))/np.abs(u));
        
        y1 = np.maximum(Ux, 0.) * u

        lamnda1 = lamnda1 +  (y1 - x)
        
        lamda0 = lamda0 + (y0-A.dot(x));
    
    return y1 * scale
        


