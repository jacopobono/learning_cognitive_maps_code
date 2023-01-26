import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
import datetime, os
import random
import matplotlib.pyplot as plt

#######
from source.utils import parameters_linear_track, run_td_lambda_new_continuousTime,run_MC
from schema import Schema, Optional, And
from types import SimpleNamespace

def fun_to_run(traj):

    traj,nr,ini,path, tlist, mode, store_data = traj
    print(nr)

    params, gamma, eta, lambda_var = parameters_linear_track(no_states=3)
    params['trajectories'] = traj

    params = update_params(mode, params, tlist, gamma, eta, state_3_rate=1)

    if mode == 'TD':
        M_stdp = run_td_lambda_new_continuousTime(**params)

    if mode == 'MC':
        M_stdp = run_MC(**params)

    if mode == 'MX':
        Trials = len(traj)
        tau = 6 #
        ratios = np.exp(-1*np.arange(Trials)/tau)
        M_stdp = np.zeros((Trials, params['no_states'], params['no_states']))
        w = []
        for trial in range(Trials):
            params, _, _, _ = parameters_linear_track(no_states=3)
            params['trajectories'] = [traj[trial]]
            params['w'] = w
            if np.random.rand()<ratios[trial]:
                params = update_params('MC', params, [tlist[trial]], gamma, eta, state_3_rate=1)
                w = run_MC(**params)
                w = w[0]
            else:
                params = update_params('TD', params, [tlist[trial]], gamma, eta, state_3_rate=1)
                w = run_td_lambda_new_continuousTime(**params)
                w = w[-1,:,:,0,0]
            M_stdp[trial] = w
            w = np.tile(np.expand_dims(np.expand_dims(w, axis=2), axis=3), [1, 1, params['N_pre_tot'], params['N_post']])


    if store_data:
        with open(path+f'/{mode}_traj_{ini+nr}.pkl','wb') as f:
            pickle.dump(M_stdp,f)

    return M_stdp

def update_params(mode, params, T_lists, gamma, eta, state_3_rate):

    if mode == 'TD':
        params['offline'] = False
        params['T_lists'] = T_lists
        params['state_3_rate'] = state_3_rate
        del params['T']
    if mode == 'MC':
        temporal_same = 2
        temporal_between = -np.log(gamma)*params['tau_plus']
        params['temporal_same'] = temporal_same
        params['temporal_between'] = temporal_between
        params['eta_stdp'] = eta/(params['A_plus']*np.exp(-temporal_same/params['tau_plus']))
        params['spike_noise_prob'] = 0.15
        params['pre_offset'] = 5
        # eta_replay = params['eta_stdp'] *params['A_plus']*np.exp(-temporal_same/params['tau_plus'])
        # gamma_replay = np.exp(-temporal_between/params['tau_plus'])
    return params

def generate_traj():
    traj = [1]
    state = 1
    while state > -1 and state < 3:
        next_state = state + random.choice([-1,1])
        traj.append(next_state)
        state = next_state
    return traj[:-1]

def root_mse(M_est, M_theor, cutoff=0):
    # trials, epochs, state, state
    mse = ((((M_est-M_theor)**2).mean(axis=0))**0.5).mean(axis=1).mean(axis=1)
    stderr = ((((M_est-M_theor)**2).mean(axis=2).mean(axis=2))**0.5).std(axis=0)/np.sqrt(len(M_est))
    return mse[cutoff:], stderr[cutoff:]


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
        Optional('state_3_T', default=100): And(int, lambda s: s in [100]),
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
        Optional('state_3_rate', default=1): And(int, lambda s: s in [1]),
        }).validate(conf)

    assert valid_conf['final_trial_nr'] > valid_conf['initial_trial_nr'], 'ERROR: Number of final trial should be larger than number of initial trial!'

    return SimpleNamespace(**valid_conf)


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

    # Define total number of trials
    tot_trials = conf.final_trial_nr-conf.initial_trial_nr

    # Define trajectory and time per state
    nr_states = 3
    traj = [ [generate_traj() for _ in  range(conf.nr_epochs)] for _ in range(tot_trials)]
    T_list = [[[100]*len(t) for t in tt] for tt in traj]

    # Define function to run and storage path
    newpath = create_path(conf)


    # Run in parallel if set
    if conf.multiprocessing:
        cc = int(conf.cpu_percentage*cpu_count())
        print(f'Using max {cc} cpus...')
        pool = Pool(cc)
        output = pool.map(lambda x: fun_to_run(x),
                          zip(traj,range(len(traj)),[conf.initial_trial_nr]*len(traj),
                                          [newpath]*len(traj), T_list,
                                          [conf.mode]*len(traj),
                                          [conf.store_data]*len(traj),
                                          ) )
        outputs=np.array(output)
    else:
        output = []
        for idx,tt in enumerate(traj):
            output.append(fun_to_run((tt,idx,conf.initial_trial_nr,
                                newpath, T_list[idx],conf.mode,
                                conf.store_data,
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
    elif conf.mode == 'MX':
        w_stdps2 = outputs

    if conf.store_data:
        pickle.dump(w_stdps2, open(newpath + '/Fig4_w_stdps2.pkl','wb'))


    T_mat = np.array(
        [[0,0.5,0,0.5],
             [0.5,0,0.5,0],
             [0,0.5,0,0.5],
             [0,0,0,1],
             ]
        )
    M_theor = np.linalg.inv(np.eye(3)-gamma_var*T_mat[:-1,:-1])

    plt.figure()
    root_mse_spiking,stderr = root_mse(w_stdps2, M_theor, 0)
    plt.plot(root_mse_spiking)
    plt.fill_between(range(len(stderr)), root_mse_spiking-stderr, root_mse_spiking+stderr,alpha=.3)


if __name__ == '__main__':


    conf_figure4 = {
        'experiment_folder': "simulations_figure4", # folder to store
        'initial_trial_nr': 1, # initial trial number
        'final_trial_nr': 3, # final trial number
        'nr_epochs': 4, # number of epochs per trial
        'cpu_percentage': 0.9, # percentage of cpus to use when running in parallel
        'multiprocessing': False, # whether to run in parallel using multiprocessing
        'store_data': False, # whether to store the experiment data
        'mode': "MC", # mode, can be MC, TD or MX
        }

    run_simulation(conf_figure4)
