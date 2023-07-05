#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:43:52 2023

@author: rezad.d.esfahani
"""

import matplotlib.pyplot as plt
from scipy.fft import (fft, ifft)
import scipy as sp
import numpy as np
import RT as rt

import SWC as sw




x  = np.arange(0, 1000, 8)
 
t, fn , f, sf, v = sw.synthetic()


d = fn[0:1000,:]#/np.max(d.reshape(-1,))

dt = 0.004

plt.figure()

plt.imshow(d, aspect='auto')




qmin = 1/500
qmax = 1/200
nq = 512


M, q = rt.DCRT(d/ sp.linalg.norm(d), x, dt, qmin, qmax, nq, mode = "ADMM", mu = 1.5, gamma = 1e-10, maxiter = 200)





#%%



plt.imshow(M, aspect='auto', extent=[0,250, q[-1], q[0] ])

plt.plot(f, v, '--r', lw = 1)

plt.xlim(0,20)
plt.ylim(200, 500)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase velocity (m/s)')
plt.title("Sparse slant stack (ADMM method)")


