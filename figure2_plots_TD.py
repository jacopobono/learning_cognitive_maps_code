import numpy as np
# import matplotlib
import pylab as plt 
from utils_learning_rules import fetch_parameters_new, theor_TD_lambda
from utils_environment import T_random_walk_linear_track
import pickle 
import os

#%% load all weights

all_weights = []


tot_runs = 10

path = 'MC_runs/Fig2_MC'
newpath = os.path.join(os.getcwd(),path)

params, gamma_var, eta, _ = fetch_parameters_new()
w_stdps = pickle.load(open(newpath+'/Fig3_w_stdps2.pkl','rb'))

# w_stdps = w_stdps2

init = np.tile(np.expand_dims(np.expand_dims(np.eye(4),0),0),[tot_runs,1,1,1])
w_stdps = np.concatenate([init,w_stdps],axis=1)

all_weights.append(w_stdps)

T_mat = T_random_walk_linear_track(params['no_states']-1)
M_theor = np.linalg.inv(np.eye(params['no_states']-1)-gamma_var*T_mat[:-1, :-1])
    
k = np.shape(w_stdps)[-1]

names = ['standard', 'time', 'rate', 'halfrate']

temporal_same = 2#.1#params['tau_plus'] * np.log(params['A_plus'])
temporal_between = -np.log(gamma_var)*params['tau_plus']
params['temporal_same'] = temporal_same
params['temporal_between'] = temporal_between#15#temporal_between

params['eta_stdp'] = eta/(params['A_plus']*np.exp(-temporal_same/params['tau_plus']))

eta_replay = params['eta_stdp'] *params['A_plus']*np.exp(-temporal_same/params['tau_plus'])
gamma_replay = np.exp(-temporal_between/params['tau_plus'])



params, gamma, eta, lambda_var = fetch_parameters_new()
M_TD = theor_TD_lambda([[0,1,2,3]]*np.shape(w_stdps)[1], gamma_replay, 1, 4, eta_replay)
# M_TD = theor_model2([[0,1,2,3]]*np.shape(w_stdps2)[1], gamma_replay, 4, eta_replay)


lw = 4
fs = 24

tdclr = 'lightseagreen'
mcclr = 'palevioletred'
mxclr = 'mediumpurple'

weights_to_plot = [ (1,1),(2,3)]
#%% weight evolution over time

for w1,w2 in weights_to_plot:
    fig = plt.figure()
    mn = w_stdps[:, :, w1, w2].mean(axis=0)
    std = np.std(w_stdps[:, :, w1, w2],axis=0)
    plt.plot(mn, color = mcclr)
    plt.fill_between(range(len(std)), mn-std, mn+std,alpha=.5, color = mcclr)
    plt.plot(M_TD[:,w1,w2],':')
    plt.ylim([0,1.4])
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('Epoch', fontsize=fs)
    plt.ylabel(f'M({w1},{w2})', fontsize=fs)
    # fig.savefig(newpath + f'/M({w1},{w2})_MC_weights.eps',bbox_inches='tight')
    
# fs = 16
    
from matplotlib import cm
# from matplotlib.colors import ListedColormap

top = cm.get_cmap('Greys', 128)
# bottom = cm.get_cmap('magma', 128*2)
# newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                        bottom(np.linspace(0, 1, 128*2))[:128]))
# newcmp = ListedColormap(newcolors, name='OrangeBlue')

fig = plt.figure()
w_stdp_matrix = plt.imshow(w_stdps[:,-1,:,:].mean(axis=0),cmap=top,vmin=0,vmax=1) 
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fs)
plt.yticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
plt.xticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
plt.xlabel('Future state (CA1)',fontsize=fs)
plt.ylabel('Current state (CA3)',fontsize=fs)
plt.title('Spiking model', fontsize=fs)
fig.savefig(newpath + '/MC_total_weights.eps',bbox_inches='tight')

fig = plt.figure()
w_stdp_matrix = plt.imshow(M_TD[-1,:,:],cmap=top,vmin=0,vmax=1) 
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fs)
plt.yticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
plt.xticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
plt.xlabel('Future state (CA1)',fontsize=fs)
plt.ylabel('Current state (CA3)',fontsize=fs)
plt.title('Theory', fontsize=fs)
fig.savefig(newpath + '/MC_theor_weights.eps',bbox_inches='tight')

#%% load all weights

all_weights = []


tot_runs = 10
for i in [0]:
    path = 'TD_runs/Fig2_data_TD'
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


params, gamma, eta, lambda_var = fetch_parameters_new()
M_TD = theor_TD_lambda([[0,1,2,3]]*np.shape(w_stdps2)[1], gamma, lambda_var, 4, eta)

#%% weight evolution over time
for w1,w2 in weights_to_plot:
    fig = plt.figure()
    mn = w_stdps2[:, :, w1, w2].mean(axis=0)
    std = np.std(w_stdps2[:, :, w1, w2],axis=0)
    plt.plot(mn, color = tdclr)
    plt.fill_between(range(len(std)), mn-std, mn+std,alpha=.5, color = tdclr)
    plt.plot(M_TD[:,w1,w2],':')
    plt.ylim([0,1.4])
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('Epoch', fontsize=fs)
    plt.ylabel(f'M({w1},{w2})', fontsize=fs)
    # fig.savefig(newpath + f'/M({w1},{w2})_TD_weights.eps',bbox_inches='tight')

fig = plt.figure()
w_stdp_matrix = plt.imshow(w_stdps2[:,-1,:,:].mean(axis=0),cmap=top,vmin=0,vmax=1) 
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fs)
plt.yticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
plt.xticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
plt.xlabel('Future state (CA1)',fontsize=fs)
plt.ylabel('Current state (CA3)',fontsize=fs)
plt.title('Spiking model', fontsize=fs)
fig.savefig(newpath + '/TD_total_weights.eps',bbox_inches='tight')

fig = plt.figure()
w_stdp_matrix = plt.imshow(M_TD[-1,:,:],cmap=top,vmin=0,vmax=1) 
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fs)
plt.yticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
plt.xticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
plt.xlabel('Future state (CA1)',fontsize=fs)
plt.ylabel('Current state (CA3)',fontsize=fs)
plt.title('Theory', fontsize=fs)
fig.savefig(newpath + '/TD_theor_weights.eps',bbox_inches='tight')
   
    
    
