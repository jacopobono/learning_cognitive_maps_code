import numpy as np
import pylab as plt
from source.utils import run_value_function,parameters_linear_track_value
import pickle
import datetime, os
from schema import Schema, Optional
from types import SimpleNamespace

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
        Optional('store_data', default=False): bool,
        Optional('time_per_state', default=100): int,
        Optional('alpha', default=0.8): float,
        Optional('A_pre', default=-11): int,
        }).validate(conf)

    valid_conf['mode'] = 'TD'
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


    # Validate cofiguration
    conf = validate_conf(conf)

    # Define function to run and storage path
    newpath = create_path(conf)

    # Define trajectory and time per state
    nr_states = 4
    traj = [list(range(nr_states))]
    T_list = [[conf.time_per_state for _ in range(nr_states)]]

    # Set parameters
    params, gamma, eta, lambda_var = parameters_linear_track_value(nr_states,
                                                                    conf.time_per_state,
                                                                    conf.alpha,
                                                                    conf.A_pre,
                                                                    )

    params['trajectories'] = traj
    params = update_params(conf.mode, params, T_list, gamma, eta)

    # Define the initial weights from CA3 to CA1
    T_mat = np.array([
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1],
        [0,0,0,0,1],
        ])
    M_theor = np.linalg.inv(np.eye(nr_states)-gamma*T_mat[:-1, :-1])
    wini = np.tile(np.expand_dims(np.expand_dims(M_theor,2),3),[1,1,params['N_pre_tot'],params['N_post']])
    params['wini'] = wini

    # Define the reward vector
    reward_vec = np.array([0,0,0,1])
    params['reward_vec'] = reward_vec

    # calculate theoretical value
    v_theor = np.matmul(M_theor, reward_vec)
    print(f'Theoretical Values: {list(v_theor)}')

    # Run spiking network
    params['step'] = params['step']*10
    reward_neuron_activity = run_value_function(**params)


    #############################
    # STORING AND PLOTTING
    #############################

    if conf.store_data:
        pickle.dump(reward_neuron_activity, open(newpath + '/rebuttal_value_function.pkl','wb'))

    rates = np.mean(np.sum(reward_neuron_activity, axis=2)/conf.time_per_state*1000, axis=1)

    # fs = 16

    fig = plt.figure()
    plt.bar(list(range(1,nr_states+1)), rates)
    if conf.store_data:
        fig.savefig(newpath+'/fig_value_function.eps',bbox_inches='tight',format='eps')

    plt.figure()
    for i in range(4):
        x = reward_neuron_activity[i]
        plt.plot(np.max(x,axis=0)*(i+1), '.', markersize=15)

if __name__ == '__main__':


    conf_value_neuron = {
        'experiment_folder': "supp_value_function", # folder to store
        'store_data': False,
        'time_per_state': 100,
        'alpha': 0.8,
        'A_pre': -11,
        }
    print(conf_value_neuron)

    run_simulation(conf_value_neuron)

