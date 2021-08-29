import pickle
from matplotlib import pylab as plt
import numpy as np

from Linear_track.utils import theor_TD_lambda


#%%

nstates=5
a = np.random.rand(nstates,nstates)
M_init = None#a*np.transpose(a)
traj = [list(range(nstates))+[nstates-1]]*50
M =theor_TD_lambda(trajectories=traj, 
                    n_states=nstates, 
                    gamma=.9, 
                    lam=0, 
                    alpha=.1, 
                    M=M_init
                    )

print(np.shape(M))

#%%

lw = 4
fs = 30

cols = ['yellow','orange','tomato','r']


fig_mse = plt.figure(figsize=(12,8))
plt.xlabel('Trials',fontsize=fs)
plt.ylabel('Weight',fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
for i in range(nstates-1):
    plt.plot(M[:,i,3], label = 'Weight '+str(i+1)+'4', linewidth = lw, color = cols[i])

plt.legend(fontsize=fs)
#fig_mse.savefig('fig3_weights.eps',bbox_inches='tight',format='eps')

plt.figure()
plt.pcolor(M[-1,:nstates-1,:nstates-1])
