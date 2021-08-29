import numpy as np
import pylab as plt 
from utils_learning_rules import fetch_parameters_new, theor_model2
from utils_environment import T_random_walk_linear_track
# from utils_plot import plot_weights_stdp_multiple_runs
from utils_nn import run_MC
import pickle 
from multiprocessing import Pool 
from main_compareMLmodel12_random_walk_parall import root_mse
import datetime, os, string


def parall_fun(traj):
    
    traj,nr,ini,path = traj
    print(nr)
    
    params, gamma_var, eta, _ = fetch_parameters_new()
    
    temporal_same = 2#.1#params['tau_plus'] * np.log(params['A_plus'])
    temporal_between = -np.log(gamma_var)*params['tau_plus']
    params['temporal_same'] = temporal_same
    params['temporal_between'] = temporal_between#15#temporal_between
    
    # print('temporal_same: ' + str(temporal_same))
    # print('temporal_between: ' + str(temporal_between))
    
    params['eta_stdp'] = eta/(params['A_plus']*np.exp(-temporal_same/params['tau_plus']))

    eta_replay = params['eta_stdp'] *params['A_plus']*np.exp(-temporal_same/params['tau_plus'])
    gamma_replay = np.exp(-temporal_between/params['tau_plus'])

    print('Eta replay: '+str(eta_replay))
    print('Gamma replay: '+str(gamma_replay))


    params['trajectories'] = traj 
    M_stdp = run_MC(**params,spike_noise_prob=0.25)#,ini='random')
    
    with open(path+f'/MC_traj_{ini+nr}.pkl','wb') as f:
        pickle.dump(M_stdp,f)
    return M_stdp#, M_evo]

if __name__ == '__main__':

    ### close all plots
    plt.close('all')
    
    ### create custom path
    newpath = str(datetime.datetime.now()) 
    newpath= newpath.translate(str.maketrans(string.punctuation, '_'*len(string.punctuation)))
    newpath=newpath.replace(" ", "_")
    newpath = 'MC_runs/'+newpath
    
    ### make directory
    os.makedirs(newpath)
    
    ### store copy of the script 
    # filename = __file__
    # filename = filename.split("/")
    # filename = filename[-1]
    # shutil.copy(filename, newpath)
      
    ### load all trajectories
    traj = pickle.load(open('1000trajectories.pkl', 'rb'))
    
    ### parameters
    ini_run = 200
    fin_run = 1000
    assert fin_run>ini_run, 'should have more than 1 run!'
    tot_runs = fin_run-ini_run
    tot_epochs = 75
    
    ### select trajectories
   
    traj = [[t[:] for t in x[:tot_epochs]] for x in traj[ini_run:fin_run]]
    #traj = [t[:tot_epochs] for t in traj[:1]]*tot_runs 
    #traj = [np.repeat(np.arange(3).reshape(-1,1).T, tot_epochs, axis=0)]
    
    ### parallel processing
    pool = Pool(4) 
    output = pool.map(parall_fun, zip(traj,range(len(traj)),[ini_run]*len(traj),[newpath]*len(traj) ) )
    # print(np.shape(output))
    output=np.array(output)
    output = output[:,1,:,:,:]
    
    ### store results
    # pickle.dump(output, open(newpath + '/output_stdp.pkl','wb'))
    
    #    
    w_stdps2 = np.array(output)#[:, 1, :, :]
    # pickle.dump(w_stdps2, open(newpath + '/w_stdps2.pkl','wb'))
    # 
    init = np.tile(np.expand_dims(np.expand_dims(np.eye(4),0),0),[tot_runs,1,1,1])
    w_stdps2 = np.concatenate([init,w_stdps2],axis=1)
    
    
    params, gamma_var, eta, _ = fetch_parameters_new()
    T_mat = T_random_walk_linear_track(params['no_states']-1)
    M_theor = np.linalg.inv(np.eye(params['no_states'])-gamma_var*T_mat)
    #
    fig = plt.figure()
    for w1 in range(params['no_states']):
        for w2 in range(params['no_states']):
            if 6 not in [w1,w2]:
                plt.plot(w_stdps2[:, :, w1, w2].mean(axis=0))
                plt.plot([0,len(w_stdps2[0,:,0,0])],[M_theor[w1, w2],M_theor[w1, w2]])
    # fig.savefig(newpath + '/weights.png')
    
    fig2 = plt.figure()
    root_mse_TD_spiking,std = root_mse(w_stdps2[:,:, :-1, :-1], M_theor[:-1, :-1], 0)
    plt.plot(root_mse_TD_spiking)   
    plt.fill_between(range(len(std)), root_mse_TD_spiking-std, root_mse_TD_spiking+std,alpha=.3)
    # fig2.savefig(newpath + '/MSE_TD_spiking.png')
    pickle.dump(root_mse_TD_spiking, open('rmse_MC_rnd_0.1.pkl','wb'))
    
    M_theor_evo = [None]*tot_runs
    for i in range(len(traj)):
        M_theor_evo[i] = theor_model2(traj[i], gamma_var, params['no_states'], eta)
    # pickle.dump(M_theor_evo, open(newpath + '/M_evo.pkl','wb'))
       
    # for w1 in range(params['no_states']):
    #     for w2 in range(params['no_states']):
    #         fig3 = plt.figure()
    #         # print(w1)
    #         # print(w2)
    #         plt.plot(w_stdps2[:, :, w1, w2].mean(axis=0), label = 'stdp')
    #         plt.plot(np.array(M_theor_evo)[:, :, w1, w2].mean(axis=0), label = 'MC')
    #         plt.plot([0,len(w_stdps2[0,:,0,0])],[M_theor[w1, w2],M_theor[w1, w2]])
    #         plt.title(f'{w1} {w2}')
    #         # fig3.savefig(newpath + '/evolution.png')
   
    
