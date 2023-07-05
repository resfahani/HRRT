
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
m[20:20+len(vec2),80] = vec2*1
m[80:80+len(vec2),50] = vec2*1

h = np.arange(128)*3

## Sampling rate
dt = 0.005

# SYNTHETIC DATA

## Maximum and Minimum velocity 
qmin = 1/500
qmax = 1/100

## Generate data using inverse RT  
d = rt.Inverse_Radon_Transform(m, h, dt, qmin, qmax)






#%% Adjoin 


# Radon transform 

M = rt.Radon_Transform(d, h, dt, qmin, qmax, nq, mode = "adj", mu = 0.1, maxiter = 10)

#% Tau-P representation

drec = rt.Inverse_Radon_Transform(M, h, dt, qmin, qmax)




fig, axs = plt.subplots(1, 3, constrained_layout=True)

ax = axs[0]

ax.imshow(d, aspect='auto',cmap ='seismic', extent=[h[0], h[-1],  dt*nt, 0 ])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')


#plt.colorbar()

ax.set_title('Data domain')


ax = axs[1]
ax.imshow(M[:,::-1], aspect='auto',cmap ='seismic', extent=[1/qmax, 1/qmin,  dt*nt, 0])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel('tau (s)')
ax.set_xlabel('velocity (m/s)')
ax.set_title('Radon (Adjoint)')

ax = axs[2]
ax.imshow(d - drec, aspect='auto',cmap ='seismic', extent=[1/qmax, 1/qmin,  dt*nt, 0])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')
ax.set_title('Residual')


plt.savefig('adj.png', dpi=500, transparent = True)



#%%  Least Square


# Radon transform 

M = rt.Radon_Transform(d, h, dt, qmin, qmax, nq, mode = "LS", mu = 0.1, maxiter = 10)

#% Tau-P representation

drec = rt.Inverse_Radon_Transform(M, h, dt, qmin, qmax)




fig, axs = plt.subplots(1, 3, constrained_layout=True)

ax = axs[0]

ax.imshow(d, aspect='auto',cmap ='seismic', extent=[h[0], h[-1],  dt*nt, 0 ], clim =[np.min(d), np.max(d)])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')


#plt.colorbar()

ax.set_title('Data domain')


ax = axs[1]
ax.imshow(M[:,::-1], aspect='auto',cmap ='seismic', extent=[1/qmax, 1/qmin,  dt*nt, 0])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel('tau (s)')
ax.set_xlabel('velocity (m/s)')
ax.set_title('Radon (LS)')

ax = axs[2]
ax.imshow(d - drec, aspect='auto',cmap ='seismic', extent=[h[0], h[-1],  dt*nt, 0 ])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')
ax.set_title('Residual')


plt.savefig('LS.png', dpi=500, transparent = True)


#%%  Iterative Re-weighted least square algorithm 


# Radon transform 

M = rt.Radon_Transform(d, h, dt, qmin, qmax, nq, mode = "IRLS", mu = 0.1, maxiter = 10)

#% Tau-P representation

drec = rt.Inverse_Radon_Transform(M, h, dt, qmin, qmax)



fig, axs = plt.subplots(1, 3, constrained_layout=True)

ax = axs[0]

ax.imshow(d, aspect='auto',cmap ='seismic', extent=[h[0], h[-1],  dt*nt, 0 ], clim =[np.min(d), np.max(d)])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')


#plt.colorbar()

ax.set_title('Data domain')


ax = axs[1]
ax.imshow(M[:,::-1], aspect='auto',cmap ='seismic', extent=[1/qmax, 1/qmin,  dt*nt, 0])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel('tau (s)')
ax.set_xlabel('velocity (m/s)')
ax.set_title('Radon domain(IRLS)')


ax = axs[2]
ax.imshow(d - drec, aspect='auto',cmap ='seismic', extent=[h[0], h[-1],  dt*nt, 0 ], clim =[np.min(d), np.max(d)])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')
ax.set_title('Residual')


plt.savefig('IRLS.png', dpi=500, transparent = True)


#%%  ADMM algorithm 


# Radon transform 

M = rt.Radon_Transform(d, h, dt, qmin, qmax, nq, mode = "ADMM", mu = 2 , gamma = 1, maxiter = 500)

#% Tau-P representation

drec = rt.Inverse_Radon_Transform(M, h, dt, qmin, qmax)



fig, axs = plt.subplots(1, 3, constrained_layout=True)

ax = axs[0]

ax.imshow(d, aspect='auto',cmap ='seismic', extent=[h[0], h[-1],  dt*nt, 0 ], clim =[np.min(d), np.max(d)])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')

#plt.colorbar()

ax.set_title('Data domain')


ax = axs[1]
ax.imshow(M[:,::-1], aspect='auto',cmap ='seismic', extent=[1/qmax, 1/qmin,  dt*nt, 0])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel('tau (s)')
ax.set_xlabel('velocity (m/s)')
ax.set_title('Radon domain (ADMM)')


ax = axs[2]
ax.imshow(d - drec, aspect='auto',cmap ='seismic', extent=[h[0], h[-1],  dt*nt, 0 ], clim =[np.min(d), np.max(d)])

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_ylabel(' Time (s)')
ax.set_xlabel('Offset')
ax.set_title('Residual')


plt.savefig('IRLS.png', dpi=500, transparent = True)

%%

a = fft(M, n = 1024, axis = -1)


plt.imshow(abs(a), aspect='auto')




#%%
