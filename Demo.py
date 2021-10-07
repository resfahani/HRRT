
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import RT as rt

from scipy import signal

#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
#lt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False


## Ricker wavelet
vec2 = signal.ricker(100, 3)

nt = 512
nq = 256
## Radon Domain, Put wavelet in the Radon domain! 
m = np.zeros([nt, nq])
m[50:50+len(vec2),100] = vec2
m[20:20+len(vec2),200] = vec2
m[80:80+len(vec2),50] = vec2

h = np.arange(128)*3

## Sampling rate
dt = 0.01

# SYNTHETIC DATA

## Maximum and Minimum velocity 
qmin = 1/500
qmax = 1/100

## Generate data using inverse RT  
d = rt.Inverse_Radon_Transform(m, h, dt, qmin, qmax)


# Radon transform 
M = rt.Radon_Transform(d, h, dt, 1/500, 1/100, nq)

#% Tau-P representation

fig, axs = plt.subplots(1, 2, constrained_layout=True)


ax = axs[0]

ax.imshow(d, aspect='auto',cmap ='coolwarm', extent=[h[0], h[-1],  dt*nt, 0 ])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')

#plt.colorbar()

ax.set_title('Data domain')


ax = axs[1]
ax.imshow(M[:,::-1], aspect='auto',cmap ='coolwarm', extent=[1/qmax, 1/qmin,  dt*nt, 0])
#plt.colorbar()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel('tau (s)')
ax.set_xlabel('velocity (m/s)')
ax.set_title('Radon domain')

#ax.set_label_position('top') 





