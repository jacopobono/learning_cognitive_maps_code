import numpy as np
import pylab as plt 
from utils_learning_rules import fetch_parameters_new
from utils_nn import run_td_lambda_new_short, run_MC
import pickle 
import multiprocessing
from multiprocessing import Pool 
import datetime, os, string

def parall_fun(traj):
    
    traj,nr,ini,path = traj
    print(nr)
    
    np.random.seed() 

    params, gamma, eta, lambda_var = fetch_parameters_new()
    params['offline'] = False
    params['w'] = np.tile(np.expand_dims(np.expand_dims(np.identity(params['no_states']), axis=2), axis=3), [1, 1, params['N_pre_tot'], params['N_post']])
    
    ###############
    
    params_MC, gamma_var, eta, _ = fetch_parameters_new()
    temporal_same = 2
    temporal_between = -np.log(gamma_var)*params_MC['tau_plus']
    params_MC['temporal_same'] = temporal_same
    params_MC['temporal_between'] = temporal_between#15#temporal_between
    params_MC['eta_stdp'] = eta/(params_MC['A_plus']*np.exp(-temporal_same/params_MC['tau_plus']))
    params_MC['w'] = np.tile(np.expand_dims(np.expand_dims(np.identity(params_MC['no_states']), axis=2), axis=3), [1, 1, params_MC['N_pre_tot'], params_MC['N_post']])
    eta_replay = params_MC['eta_stdp'] *params_MC['A_plus']*np.exp(-temporal_same/params_MC['tau_plus'])
    gamma_replay = np.exp(-temporal_between/params_MC['tau_plus'])
    print('Eta replay: '+str(eta_replay))
    print('Gamma replay: '+str(gamma_replay))

    ###############
    
    Trials = len(traj)#*2 
    
    #tau = 6 #
    #ratios = np.exp(-1*np.arange(Trials)/tau)

    w_store = np.zeros((Trials, params['no_states'], params['no_states']))
    
    # seq_updates = np.zeros((Trials)) #0 = TD, 1 = MC
    
    for trial in range(Trials):
        params_MC['trajectories'] = [traj[trial]]
        if np.random.rand()<0.5:#ratios[trial]:
            _, w = run_MC(**params_MC)
            # print(np.shape(w))
        else:
            params['trajectory'] = traj[trial]
            w = run_td_lambda_new_short(**params)
            # print(np.shape(w))
        w_store[trial] = w[0]
        w = np.tile(np.expand_dims(np.expand_dims(w[0], axis=2), axis=3), [1, 1, params['N_pre_tot'], params['N_post']])
        params_MC['w'] = w
        params['w'] = w
        # seq_updates[trial] = 1

    #M_TD = theor_TD_lambda(traj, gamma, lambda_var, params['no_states'], eta)
    # M_mix = mix_model12_seq(traj, gamma, lambda_var, params['no_states'], eta, seq_updates)
    output = w_store #{'M': M_mix, 'w_stdp':w_store}
    
    with open(path+f'/MX_traj_{ini+nr}.pkl','wb') as f:
        pickle.dump(w_store,f)
        
    return output

if __name__ == '__main__':

        
    plt.close('all')
    
    newpath = str(datetime.datetime.now()) 
    newpath= newpath.translate(str.maketrans(string.punctuation, '_'*len(string.punctuation)))
    newpath=newpath.replace(" ", "_")
    newpath = 'MX_runs/'+newpath
    
    ### make directory
    os.makedirs(newpath)
    
    traj = pickle.load(open('1000trajectories.pkl', 'rb'))
    
    ini_run = 0
    fin_run = 1000
    assert fin_run>ini_run, 'should have more than 1 run!'
    tot_runs = fin_run-ini_run
    tot_epochs = 75
    
    
    traj = [[t[:] for t in x[:tot_epochs]] for x in traj[ini_run:fin_run]]
    #traj = [t[:] for t in traj[:1][:tot_epochs]]*tot_runs 
    
    cc = int(.9*multiprocessing.cpu_count())
    pool = Pool(cc)   
    output = pool.map(parall_fun, zip(traj,range(len(traj)),[ini_run]*len(traj),[newpath]*len(traj) ) )
    
    
    output = np.array(output)
    
    
    params, gamma_var, _, _ = fetch_parameters_new()
#    w_stdps2 = np.mean(np.sum(w_stdps[:,:,:,:,:,:],axis=-2)/params['N_pre'],axis=-1)
#    pickle.dump(w_stdps2, open(newpath + '/w_stdps2.pkl','wb'))
#    
#    T_mat = T_random_walk_linear_track(params['no_states']-1)
#    M_theor = np.linalg.inv(np.eye(params['no_states']-1)-gamma_var*T_mat[:-1, :-1])
#    #
    fig = plt.figure()
    for w1 in range(params['no_states']-1):
        for w2 in range(params['no_states']-1):
            plt.plot(output[:, :, w1, w2].mean(axis=0))
            # plt.plot(M[:, :, w1, w2].mean(axis=0))
    # fig.savefig(newpath + '/match.png')
#    
#    fig2 = plt.figure()
#    root_mse_TD_spiking = root_mse(w_stdps2[:,:,:-1,:-1], M_theor, 0)
#    plt.plot(root_mse_TD_spiking)    
#    fig2.savefig(newpath + '/MSE_TD_spiking.png')
#    
#    fig3 = plt.figure()
#    w_stdp_matrix = plt.imshow(w_stdps2[:,-1,:,:].mean(axis=0)) 
#    fig3.savefig(newpath + '/weights_matrix.png')
