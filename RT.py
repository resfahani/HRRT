
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt




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
    
    M = np.fft.fft(m, n = nfft, axis = 0)
    
    
    ilow = 0
    ihigh = nfft//2

    D = np.zeros([ihigh, nx], dtype=np.complex128)
    f = 2 * np.pi /nfft/dt

    op = np.exp(1j * f * h[:,np.newaxis ] @ q[np.newaxis,:])

    op = np.conj(op)
    L = np.ones([nx, nq])

    for ifreq in range(ilow, ihigh):
        
        p = M[ifreq,:][:,np.newaxis]
        p = (p)
        y = (L @ p)[:,0]
        D[ifreq,:] = (y)   
        #D[nfft-1- ifreq,:] = np.conj(y)
        L = L  * op


    d = 2 * np.fft.ifft(D ,n = 1 * nfft ,axis = 0)
    d = np.real(d)
    d =d[:nt ,:]
    
    return d


def Radon_Transform(d, h, dt, qmin, qmax, nq):
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

    Returns
    -------
    m : Radon domain (tau-nq)
        DESCRIPTION.

    """
    nt, nx = np.shape(d)
    q = np.linspace(qmin, qmax, nq)    

    nfft = 1 * next_power_of_2(nt)
    D = np.fft.fft(d, n = nfft, axis = 0)
    
    ilow = 0
    ihigh = nfft//2

    M = np.zeros([nfft, nq], dtype=np.complex128)
    
    f = 2 * np.pi /nfft/dt

    op = np.exp(1j * f * h[:,np.newaxis ] @ q[np.newaxis,:])

    op = np.conj(op)
    
    L = np.ones([nx, nq])

    for ifreq in range(ilow, ihigh):
        p = D[ifreq,:][:,np.newaxis]
        p = np.conj(p)
        y = ((L.T) @ p)[:,0]
        
        M[ifreq,:] = np.conj(y)
    
        L = L  * op

    m = 2 * np.fft.ifft(M ,n = 1 * nfft ,axis = 0)
    m = np.real(m)
    m =m[:nt ,:]
    
    return m
    


def PhVelEst(m, dt):
    nt, nq = np.shape(m)
    nfft = 2 * next_power_of_2(nt)
    M = np.fft.fft(m, n = nfft, axis = 0)
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


