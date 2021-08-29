import pickle
import sys

sys.path.append('/Users/jac2/Documents/Successor_paper/SR_code')
from utils_learning_rules import fetch_parameters_new
from main_compareMLmodel12_random_walk_parall import root_mse
from utils_environment import T_random_walk_linear_track
from matplotlib import pylab as plt
import numpy as np

#from utils_learning_rules import theor_model2
plt.close('all') 

# MC_folder = '2019_09_18_20_26_21_874424'
# TD_folder = 'td_random_walk'
# mix_folder = '2019_09_20_16_50_39_735852'

# Plot all RMSEs 
# mse_mc = pickle.load(open(MC_folder + '/MSE_MC.pkl', 'rb'))
# mse_mc_stdp = pickle.load(open(MC_folder + '/MSE_MC_stdp.pkl', 'rb'))

# mse_td = pickle.load(open(TD_folder + '/MSE_TD.pkl', 'rb'))
# mse_td_stdp = pickle.load(open(TD_folder + '/MSE_TD_stdp.pkl', 'rb'))

# mse_mix = pickle.load(open(mix_folder + '/MSE_TD.pkl', 'rb'))
# mse_mix_stdp = pickle.load(open(mix_folder + '/MSE_TD_stdp.pkl', 'rb'))

mse_td_stdp = pickle.load(open('data/rmse_TD.pkl', 'rb'))
mse_mc_stdp = pickle.load(open('data/rmse_MC.15.pkl', 'rb'))
# mse_mx_stdp = pickle.load(open('data/rmse_MX.pkl', 'rb'))
mse_mx_stdp = pickle.load(open('data/rmse_MX_fifty_fifty.pkl', 'rb'))




# all traces
# dimensions: repeats, maze_runs, states, states
# TD_all_traces = pickle.load(open(TD_folder + '/output_stdp.pkl', 'rb'))
# MC_all_traces = pickle.load(open(MC_folder + '/weights_version.pkl', 'rb'))
# MX_all_traces = pickle.load(open(mix_folder + '/output_stdp.pkl', 'rb'))

MC_all_traces = pickle.load(open('data/alltraces_MC.15_tempsame2.pkl', 'rb'))
TD_all_traces = pickle.load(open('data/alltraces_TD.pkl', 'rb'))
# MX_all_traces = pickle.load(open('data/alltraces_MX.pkl', 'rb'))
MX_all_traces = pickle.load(open('data/alltraces_MX_fifty_fifty.pkl', 'rb'))

#%% calculate rmse

params, gamma_var, _, _ = fetch_parameters_new()
T_mat = T_random_walk_linear_track(params['no_states']-1)
M_theor = np.linalg.inv(np.eye(params['no_states']-1)-gamma_var*T_mat[:-1, :-1])
    
mse_mx_stdp, std = root_mse(MX_all_traces[:, :, :-1,:-1], M_theor, 0)
# pickle.dump(mse_mx_stdp, open('data/rmse_MX_fifty_fifty.pkl','wb'))


#%%

lw = 4
fs = 30

tdclr = 'lightseagreen'
mcclr = 'palevioletred'
mxclr = 'mediumpurple'


fig_mse = plt.figure(figsize=(12,8))
plt.xlabel('Trials',fontsize=fs)
plt.ylabel('RMSE',fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.plot(mse_td_stdp, label = 'TD stdp', linewidth = lw, color = tdclr)
plt.plot(mse_mc_stdp, label = 'MC stdp', linewidth = lw, color = mcclr)
plt.plot(mse_mx_stdp, label = 'Mix stdp', linewidth = lw, color = mxclr)

plt.legend(fontsize=fs)
# fig_mse.savefig('fig4_MSE_stdp_fifty_fifty.eps',bbox_inches='tight',format='eps')

T_mat = np.array([[0,0.5,0],[.5,0,.5],[0,.5,0]])
theor = np.linalg.inv(np.eye(3)-0.9263892959738923*T_mat[:, :]) #0.9263892959738923
xvals = range(75)


variance_TD = []
variance_MC = []


nsamples = 3
w_to_plot = (1,0) ### Which weight we choose for the individual traces
for name,data,clr in zip(['TD','MC','MX'],[TD_all_traces,MC_all_traces,MX_all_traces],[tdclr,mcclr,mxclr]):
    stds = np.std(data[:,xvals,w_to_plot[0],w_to_plot[1]],axis=0)
    mns = np.mean(data[:,xvals,w_to_plot[0],w_to_plot[1]],axis=0)
    plt.figure(figsize=(12,8))
    plt.fill_between(xvals, mns-stds, mns+stds, alpha=.5, color=clr)
    plt.plot(xvals,mns, color=clr, linewidth = lw)
    plt.plot([xvals[0],xvals[-1]],[theor[w_to_plot[0],w_to_plot[1]],theor[w_to_plot[0],w_to_plot[1]]],color='k')
    for j in np.random.choice(list(range(100)),nsamples,replace=False): #range(nsamples):
        plt.plot(data[j,xvals,w_to_plot[0],w_to_plot[1]], color=clr, linewidth = lw)
    plt.ylim([0,2])
    plt.xlabel('Trials',fontsize=fs)
    plt.ylabel('Weight',fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('Weight ' + str(w_to_plot) + ', '+name+' stdp',fontsize=fs)
    plt.savefig('fig4_'+name+'.eps',bbox_inches='tight',format='eps')
    print(name+' stdp: '+str(np.sum(stds)))
    

