import numpy as np
import pylab as plt
from source.utils import parameters_linear_track, run_td_lambda_new_continuousTime, run_MC
import pickle
import datetime, os
from multiprocessing import Pool, cpu_count
from schema import Schema, Optional, And
from types import SimpleNamespace
from matplotlib import cm
from matplotlib.colors import ListedColormap


def fun_to_run(traj):

    traj,nr,ini,path,T_lists,mode,store,nr_states = traj
    # params, gamma, eta, lambda_var = fetch_parameters_new()
    params, gamma, eta, lambda_var = parameters_linear_track(nr_states)

    params['trajectories'] = traj
    params = update_params(mode, params, T_lists, gamma, eta)

    if mode == 'TD':
        M_stdp = run_td_lambda_new_continuousTime(**params)

    if mode == 'MC':
        M_stdp = run_MC(**params)
    if store:
        with open(path+f'/{mode}_traj_{ini+nr}.pkl','wb') as f:
            pickle.dump(M_stdp,f)

    return M_stdp

def update_params(mode, params, T_lists, gamma, eta):

    if mode == 'TD':
        params['offline'] = False
        params['T_lists'] = T_lists
        del params['T']
    if mode == 'MC':
        temporal_same = 2
        temporal_between = -np.log(gamma)*params['tau_plus']
        params['temporal_same'] = temporal_same
        params['temporal_between'] = temporal_between
        params['eta_stdp'] = eta/(params['A_plus']*np.exp(-temporal_same/params['tau_plus']))
        params['spike_noise_prob'] = 0.2
        params['pre_offset'] = 5
        # eta_replay = params['eta_stdp'] *params['A_plus']*np.exp(-temporal_same/params['tau_plus'])
        # gamma_replay = np.exp(-temporal_between/params['tau_plus'])
    return params

def validate_conf(conf: dict) -> SimpleNamespace :
    """
    Validate the hyperparameters in the dictionary.

    Parameters
    ----------
    conf : dict
        Dictionary containing hyperparemeters to run the simulation.

    Returns
    -------
    SimpleNamespace
        Namespace containing the validated hyperparameters.

    """
    valid_conf = Schema({
        Optional('experiment_folder', default="simulation"): str,
        Optional('initial_trial_nr', default=1): And(int, lambda s: s>0),
        Optional('final_trial_nr', default=2): And(int, lambda s: s>0),
        Optional('nr_epochs', default=2): And(int, lambda s: s>0),
        # Optional('nr_states', default=10): And(int, lambda s: s>2),
        Optional('cpu_percentage', default=0.9): And(float, lambda s: s>0,
                                                     lambda s: s<=1),
        Optional('multiprocessing', default=False): bool,
        Optional('store_data', default=False): bool,
        Optional('mode', default="MC"): And(str, lambda s: s in ['TD', 'MC', 'MX']),
        }).validate(conf)

    assert valid_conf['final_trial_nr'] > valid_conf['initial_trial_nr'], 'ERROR: Number of final trial should be larger than number of initial trial!'

    return SimpleNamespace(**valid_conf)

def create_path(conf: dict):
    """
    Create experiment path.

    Parameters
    ----------
    conf: dict
        Dict of the yaml configuration file.
    logg: LOGG
        Logger object.

    Returns
    -------
    newpath: str
        Path to experiment folder.
    """

    # Create experiment folder
    date_time = str(datetime.datetime.now()).split('.')[0].replace('-','_').replace(':','_').replace(' ','_')
    newpath = os.path.join(conf.experiment_folder, date_time)

    newpath = f'{conf.mode}_runs/'+newpath

    # Create storage folder
    if conf.store_data:
        os.makedirs(newpath)
        print('Folder {} created!'.format(newpath))

    return newpath

def generate_2d_trajectories(nr_trajectories):
    init_state = 16
    reward_state = 1
    next_states = {1: [17],
                   2: [1,3,6],
                   3: [2,4,7],
                   4: [3,8],
                   5: [1,6,9],
                   6: [2,5,7,10],
                   7: [3,6,8],
                   8: [4,7,12],
                   9: [5,10,13],
                   10: [6,9,14],
                   12: [8,16],
                   13: [9,14],
                   14: [10,13,15],
                   15: [14,16],
                   16: [12,15],
                   }

    all_trajectories = []

    for _ in range(nr_trajectories):
        traj = []
        current_state = init_state
        while current_state != reward_state:
            traj.append(current_state-1)
            current_state = np.random.choice(next_states[current_state])
        traj.append(reward_state-1)
        # traj.append
        all_trajectories.append(traj)

    return all_trajectories




def run_simulation(conf):
    """
    Run a simulation on the linear track.

    Parameters
    ----------
    """


    # validate cofiguration
    conf = validate_conf(conf)

    # Define total number of trials
    tot_trials = conf.final_trial_nr-conf.initial_trial_nr

    # Define trajectory and time per state
    nr_states=17
    traj = [generate_2d_trajectories(conf.nr_epochs) for _ in range(tot_trials)]
    T_list = [[[100 for _ in y] for y in x] for x in traj]

    # Define function to run and storage path
    newpath = create_path(conf)


    # Run in parallel if set
    if conf.multiprocessing:
        cc = int(conf.cpu_percentage*cpu_count())
        print(f'Using max {cc} cpus...')
        pool = Pool(cc)
        output = pool.map(fun_to_run, ## DON'T USE LAMBDA FUNCTION IN MULTIPROCESSING
                          zip(traj,range(len(traj)),[conf.initial_trial_nr]*len(traj),
                                          [newpath]*len(traj), T_list,
                                          [conf.mode]*len(traj),
                                          [conf.store_data]*len(traj),
                                          [nr_states]*len(traj),
                                          )
                          )
        outputs=np.array(output)
    else:
        output = []
        for idx,tt in enumerate(traj):
            output.append(fun_to_run((tt,idx,conf.initial_trial_nr,
                                newpath, np.array(T_list),conf.mode,
                                conf.store_data,
                                nr_states,
                                )
                                )
                          )
        outputs=np.array(output)


    # cumulative lengths
    l = [np.cumsum([len(x) for x in t]) for t in traj]
    # params, gamma_var, _, _ = fetch_parameters_new()
    params, gamma_var, _, _ = parameters_linear_track(nr_states)
    if conf.mode == 'TD':
        w_stdps = np.array([[outputs[i][l_i] for l_i in l[i]] for i in range(len(outputs))])
        w_stdps2 = np.mean(np.sum(w_stdps[:,:,:,:,:,:],axis=-2)/params['N_pre'],axis=-1)
    elif conf.mode == 'MC':
        w_stdps2 = outputs #np.array(outputs)


    #############################
    # STORING AND PLOTTING
    #############################

    if conf.store_data:
        pickle.dump(w_stdps2, open(newpath + f'/rebuttal_2d_env_{conf.mode}_w_stdps2.pkl','wb'))


    init = np.tile(np.expand_dims(np.expand_dims(np.eye(nr_states),0),0),[len(traj),1,1,1])
    w_stdps2 = np.concatenate([init,w_stdps2],axis=1)

    fs = 16

    top = cm.get_cmap('Greys', 128)
    bottom = cm.get_cmap('magma', 128*2)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128*2))[:128]))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    fig3 = plt.figure()
    plt.imshow(w_stdps2[:,-2,:-1,:-1].mean(axis=0),cmap=newcmp,vmin=0,vmax=2)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fs)
    tcks = list(range(nr_states-1))
    plt.yticks(ticks=tcks,labels=[x+1 for x in tcks],fontsize=fs)
    plt.xticks(ticks=tcks,labels=[x+1 for x in tcks],fontsize=fs)
    plt.xlabel('Future state (CA1)',fontsize=fs)
    plt.ylabel('Current state (CA3)',fontsize=fs)
    plt.title(f'{conf.mode}',fontsize=fs)
    if conf.store_data:
        fig3.savefig(newpath+f'/fig_rebuttal_2d_env_{conf.mode}.eps',bbox_inches='tight',format='eps')

    T_mat = np.array([[0]*16+[1],
             [1/3, 0, 1/3, 0, 0, 1/3] + [0]*11,
             [0,1/3, 0, 1/3, 0, 0, 1/3] + [0]*10,
             [0,0,1/2, 0, 0, 0, 0, 1/2] + [0]*9,
             [1/3] + [0]*4 + [1/3, 0, 0, 1/3] + [0]*8,
             [0,1/4] + [0]*2 + [1/4, 0, 1/4, 0, 0, 1/4] + [0]*7,
             [0,0,1/3, 0, 0, 1/3, 0, 1/3] + [0]*9,
             [0,0,0,1/3, 0, 0, 1/3, 0, 0,0,0,1/3] + [0]*5,
             [0,0,0,0,1/3, 0, 0, 0, 0, 1/3,0,0,1/3] + [0]*4,
             [0,0,0,0,0,1/3, 0, 0, 1/3, 0, 0,0,0,1/3] + [0]*3,
             [0]*16+[1],
             [0]*7+[1/2]+[0]*7+[1/2]+[0],
             [0]*8+[1/2]+[0]*4+[1/2]+[0]*3,
             [0]*9+[1/3]+[0]*2+[1/3]+[0]*1+[1/3]+[0]*2,
             [0]*13+[1/2]+[0]*1+[1/2]+[0],
             [0]*11+[1/2]+[0]*2+[1/2]+[0]*2,
             [0]*16+[1],
             ])

    M_theor = np.linalg.inv(np.eye(nr_states-1)-gamma_var*T_mat[:-1, :-1])

    fig4 = plt.figure()
    plt.imshow(M_theor,cmap=newcmp,vmin=0,vmax=2)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fs)
    tcks = list(range(nr_states-1))
    plt.yticks(ticks=tcks,labels=[x+1 for x in tcks],fontsize=fs)
    plt.xticks(ticks=tcks,labels=[x+1 for x in tcks],fontsize=fs)
    plt.xlabel('Future state (CA1)',fontsize=fs)
    plt.ylabel('Current state (CA3)',fontsize=fs)
    plt.title('Ground Truth',fontsize=fs)
    if conf.store_data:
        fig4.savefig(newpath+'/theory.eps',bbox_inches='tight',format='eps')


if __name__ == '__main__':


    conf_supp_2d_env = {
        'experiment_folder': "conf_supp_2d_env", # folder to store
        'initial_trial_nr': 1, # initial trial number
        'final_trial_nr': 25, # final trial number #25 for mc, 5 for td
        'nr_epochs': 20, # number of epochs per trial
        'cpu_percentage': 0.7, # percentage of cpus to use when running in parallel
        'multiprocessing': True, # whether to run in parallel using multiprocessing
        'store_data': True, # whether to store the experiment data
        'mode': "MC", # mode, can be MC, TD or MX
        }

    run_simulation(conf_supp_2d_env)

