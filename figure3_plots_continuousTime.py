import numpy as np
# import matplotlib
import pylab as plt 
from utils_learning_rules import fetch_parameters_new
from utils_environment import T_random_walk_linear_track
import pickle 
import os

#%% load all weights

all_weights = []


tot_runs = 10
for i in range(4):
    path = f'TD_runs/Fig3_data_0{i}'
    newpath = os.path.join(os.getcwd(),path)

    params, gamma_var, _, _ = fetch_parameters_new()
    w_stdps2 = pickle.load(open(newpath+'/Fig3_w_stdps2.pkl','rb'))
    
    init = np.tile(np.expand_dims(np.expand_dims(np.eye(4),0),0),[tot_runs,1,1,1])
    w_stdps2 = np.concatenate([init,w_stdps2],axis=1)
    
    all_weights.append(w_stdps2)
    
    T_mat = T_random_walk_linear_track(params['no_states']-1)
    M_theor = np.linalg.inv(np.eye(params['no_states']-1)-gamma_var*T_mat[:-1, :-1])
    
k = np.shape(w_stdps2)[-1]

names = ['standard', 'time', 'rate', 'halfrate']

#%% weight evolution over time
fig = plt.figure()
for w1 in range(params['no_states']-1):
    for w2 in range(params['no_states']-1):
        if 3 not in [w1,w2]:
            plt.plot(w_stdps2[:, :, w1, w2].mean(axis=0))
            # plt.plot([0,len(w_stdps2[0,:,0,0])],[M_theor[w1, w2],M_theor[w1, w2]])
# fig.savefig(newpath + '/weights.png')
    
#%% final weight matrix    
fs = 16

from matplotlib import cm
from matplotlib.colors import ListedColormap

for jj in range(k):
    ww = all_weights[jj]
    top = cm.get_cmap('Greys', 128)
    bottom = cm.get_cmap('magma', 128*2)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128*2))[:128]))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    
    fig3 = plt.figure()
    w_stdp_matrix = plt.imshow(ww[:,-2,:,:].mean(axis=0),cmap=newcmp,vmin=0,vmax=2) 
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fs)
    plt.yticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
    plt.xticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
    plt.xlabel('Future state (CA1)',fontsize=fs)
    plt.ylabel('Current state (CA3)',fontsize=fs)
    plt.title(names[jj])
# fig3.savefig('fig3_.eps',bbox_inches='tight',format='eps')
    
#%% place fields
fs = 16
lw = 6
num_sims = 3
from scipy import interpolate
# import seaborn as sn
fig,axs = plt.subplots(2,2, sharex=True,  figsize=(8,8))
for i in range(k): 
    for jj in range(num_sims):
        ww = all_weights[jj]
        data = ww[:,-1,:,i].mean(axis=0) # only columns, mean over 10 runs
        # plt.figure()
        # plt.bar(x = list(range(1,k+1)), height=data,alpha=.5,label=names[jj])
        axs[i//2,i%2].scatter(x = list(range(1,k+1)), y=data,alpha=.5)#,label=names[jj])
        x = list(range(0,k+2))
        y = np.concatenate([[0],data,[0]])
        f2 = interpolate.interp1d(x, y, kind='quadratic')
        xnew = np.linspace(0,k+1,51)
        axs[i//2,i%2].plot(xnew[:10*(i+2)], f2(xnew)[:10*(i+2)],alpha=.3,label=names[jj],linewidth=lw)
        axs[i//2,i%2].set_xticks(list(range(1,k+1)))
        axs[i//2,i%2].set_xticklabels(list(range(1,k+1)), fontsize=fs)
        
        axs[i//2,i%2].set_yticks(np.arange(0,2.1,.5))
        axs[i//2,i%2].set_yticklabels(np.arange(0,2.1,.5), fontsize=fs)
        
        # plt.ylim([0,2.5])
        # plt.fill_between(xnew[:10*(i+2)], 0, f2(xnew)[:10*(i+2)],alpha=.3,label=names[jj])
    
plt.xlim([0,5])
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels,fontsize=fs,)# loc='upper right')
plt.yticks(fontsize=fs)
fig.text(0.04, 0.5, 'Firing rate', va='center', rotation='vertical', fontsize=fs)
fig.text(0.5, 0.05, 'State', ha='center', fontsize=fs)
# plt.xlabel('State', fontsize=fs)

plt.setp(axs[0,0].get_xticklabels(), visible=False)
plt.setp(axs[0,1].get_xticklabels(), visible=False)
plt.setp(axs[0,1].get_yticklabels(), visible=False)
plt.setp(axs[1,1].get_yticklabels(), visible=False)
# fig.savefig('fig3_placefileds.eps',bbox_inches='tight',format='eps')    
    
    # for ax in axs.flat:
    #     ax.label_outer()
    
    
