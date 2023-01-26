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


def fun_to_run(traj, nr_states=4):

    traj,nr,ini,path,T_lists,mode,store = traj
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

    params['ini'] = 'random'
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
        Optional('state_3_T', default=100): And(int, lambda s: s in [100,200]),
        Optional('experiment_folder', default="simulation"): str,
        Optional('initial_trial_nr', default=1): And(int, lambda s: s>0),
        Optional('final_trial_nr', default=2): And(int, lambda s: s>0),
        Optional('nr_epochs', default=2): And(int, lambda s: s>0),
        # Optional('nr_states', default=10): And(int, lambda s: s>2),
        Optional('cpu_percentage', default=0.9): And(float, lambda s: s>0,
                                                     lambda s: s<=1),
        Optional('multiprocessing', default=False): bool,
        Optional('store_data', default=False): bool,
        Optional('mode', default="MC"): And(str, lambda s: s in ['TD', 'MC']),
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


def run_simulation(conf):
    """
    Run a simulation on the linear track.

    Parameters
    ----------
    """
    absorbing_state = False

    # validate cofiguration
    conf = validate_conf(conf)

    # Define total number of trials
    tot_trials = conf.final_trial_nr-conf.initial_trial_nr

    # Define trajectory and time per state
    if absorbing_state:
        nr_states = 5
        traj = np.repeat([np.repeat([list(range(nr_states)) + [nr_states-1]], conf.nr_epochs, axis=0)], tot_trials, axis=0)
        T_list = [[100]*nr_states + [100]]*conf.nr_epochs
    else:
        nr_states = 4
        traj = np.repeat([np.repeat([list(range(nr_states))], conf.nr_epochs, axis=0)], tot_trials, axis=0)
        T_list = [[100]*nr_states ]*conf.nr_epochs

    # Define function to run and storage path
    newpath = create_path(conf)


    # Run in parallel if set
    if conf.multiprocessing:
        cc = int(conf.cpu_percentage*cpu_count())
        print(f'Using max {cc} cpus...')
        pool = Pool(cc)
        output = pool.map(lambda x: fun_to_run(x, nr_states=nr_states),
                          zip(traj,range(len(traj)),[conf.initial_trial_nr]*len(traj),
                                          [newpath]*len(traj), [np.array(T_list)]*len(traj),
                                          [conf.mode]*len(traj),
                                          [conf.store_data]*len(traj),
                                          ) )
        outputs=np.array(output)
    else:
        output = []
        for idx,tt in enumerate(traj):
            output.append(fun_to_run((tt,idx,conf.initial_trial_nr,
                                newpath, np.array(T_list),conf.mode,
                                conf.store_data,
                                ),
                                nr_states=nr_states)
                          )
        outputs=np.array(output)


    # cumulative lengths
    l = [np.cumsum([len(x) for x in t]) for t in traj]
    # params, gamma_var, _, _ = fetch_parameters_new()
    params, gamma_var, _, _ = parameters_linear_track(nr_states)
    if conf.mode == 'TD':
        w_stdps = np.array([[outputs[i][l_i] for l_i in l[i]] for i in range(len(outputs))])
        w_stdps2 = np.mean(np.sum(w_stdps[:,:,:,:,:,:],axis=-2)/params['N_pre'],axis=-1)
        # w_stdps2 = outputs #np.expand_dims(outputs, axis=0)
    elif conf.mode == 'MC':
        w_stdps2 = outputs #np.array(outputs)


    #############################
    # STORING AND PLOTTING
    #############################

    if conf.store_data:
        pickle.dump(w_stdps2, open(newpath + '/rebuttal_random_init_w_stdps2.pkl','wb'))


    # init = np.tile(np.expand_dims(np.expand_dims(np.eye(4),0),0),[tot_trials,1,1,1])
    # w_stdps2 = np.concatenate([init,w_stdps2],axis=1)

    fs = 16

    top = cm.get_cmap('Greys', 128)
    bottom = cm.get_cmap('magma', 128*2)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128*2))[:128]))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    fig3 = plt.figure()
    plt.imshow(w_stdps2[:,-2,:,:].mean(axis=0),cmap=newcmp,vmin=0,vmax=2)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fs)
    plt.yticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
    plt.xticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
    plt.xlabel('Future state (CA1)',fontsize=fs)
    plt.ylabel('Current state (CA3)',fontsize=fs)
    if conf.store_data:
        fig3.savefig(newpath+'/fig_sr.eps',bbox_inches='tight',format='eps')

    pre_state = 2
    post_state = 3
    fig4 = plt.figure()
    plt.plot(w_stdps2[:,:,pre_state,post_state].mean(axis=0))
    cbar.ax.tick_params(labelsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.xlabel('Time',fontsize=fs)
    plt.ylabel('Weight',fontsize=fs)
    if conf.store_data:
        fig4.savefig(newpath+f'/fig_{pre_state}_{post_state}.eps',bbox_inches='tight',format='eps')


if __name__ == '__main__':

    conf_supp_RandomInit = {
        'state_3_T': 100, # time in state 3
        'experiment_folder': "supp_RandomInit", # folder to store
        'initial_trial_nr': 1, # initial trial number
        'final_trial_nr': 2, # final trial number
        'nr_epochs': 300, # number of epochs per trial
        'cpu_percentage': 0.9, # percentage of cpus to use when running in parallel
        'multiprocessing': False, # whether to run in parallel using multiprocessing
        'store_data': False, # whether to store the experiment data
        'mode': "MC", # mode, can be MC, TD
        }

    run_simulation(conf_supp_RandomInit)

