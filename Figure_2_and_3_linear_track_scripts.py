import numpy as np
import pylab as plt 
from utils_learning_rules import fetch_parameters_new
from utils_environment import T_random_walk_linear_track
from utils_nn import run_td_lambda_new_continuousTime, run_MC
import pickle 
import multiprocessing
from multiprocessing import Pool 
from main_compareMLmodel12_random_walk_parall import root_mse
import datetime, os, string

import os
import datetime
import pickle
from multiprocessing import Pool, cpu_count
import numpy as np
import yaml
from utils import theor_TD_lambda, parameters_long_linear_track, run_spiking_td_lambda
import logging
from fire import Fire
from schema import Schema, Optional, And
from types import SimpleNamespace


def parall_fun(traj):
    
    traj,nr,ini,path,T_lists = traj
    print(nr)
    
    params, gamma, eta, lambda_var = fetch_parameters_new()
    params['trajectories'] = traj 
    params['offline'] = False
    params['T_lists'] = T_lists
    del params['T']
    # print(gamma)
    M_stdp = run_td_lambda_new_continuousTime(**params)#,ini='random')    
    # M_TD = theor_TD_lambda(traj, gamma, lambda_var, params['no_states'], eta)

    with open(path+f'/TD_traj_{ini+nr}.pkl','wb') as f:
        pickle.dump(M_stdp,f)
        
    return M_stdp, np.zeros_like(M_stdp) #M_TD

def parall_fun_MC(traj):
    
    traj,nr,ini,path,T_lists = traj
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

    # print('Eta replay: '+str(eta_replay))
    # print('Gamma replay: '+str(gamma_replay))


    params['trajectories'] = traj 
    _,M_stdp = run_MC(**params,spike_noise_prob=0.2)#,ini='random')
    
    # M_theor, theor_TD_lambda(traj, gamma_replay, 1, 4, eta_replay, M_TD_lambda=[])
    
    with open(path+f'/MC_traj_{ini+nr}.pkl','wb') as f:
        pickle.dump(M_stdp,f)
    # with open(path+f'/MCtheor_traj_{ini+nr}.pkl','wb') as f:
    #     pickle.dump(M_theor,f)
        
    return M_stdp, np.zeros_like(M_stdp)
    
def validate_yaml(conf: dict) -> SimpleNamespace :
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
        Optional('tot_runs', default=1): And(int, lambda s: s>0),
        Optional('tot_epochs', default=2): And(int, lambda s: s>0),
        Optional('n_states', default=10): And(int, lambda s: s>2),
        Optional('cpu_percentage', default=0.9): And(float, lambda s: s>0,
                                                     lambda s: s<=1),
        Optional('logger_level', default=10): And(int, lambda s: s>=0),
        Optional('logger_file', default=""): str,
        Optional('multiprocessing', default=False): bool,
        Optional('store_data', default=False): bool,
        }).validate(conf)
    
    return SimpleNamespace(**valid_conf)

class LOGG:
    """
    Create logger.
    
    Create logger with streamhandler. Method to add filehandler.
    """
    
    def __init__(self,logger_level: int = logging.DEBUG):
        """
        Set logger.
        
        Parameters
        ----------
        logger_level: int. Defaults to logging.DEBUG = 10
            Create logger object, set formatting and level.
        """
        #logger object
        self.logger = logging.getLogger('main_logger')
        self.formatter = logging.Formatter('%(funcName)s :: %(levelname)s :: %(message)s')
        self.logger.setLevel(logging.DEBUG)
        
        #logger stream handler
        sh = logging.StreamHandler()
        sh.setLevel(logger_level)
        sh.setFormatter(self.formatter)
        
        #add handler
        self.logger.addHandler(sh)
    
    def add_file_handler(
            self,
            logger_path: str, 
            logger_level: int = logging.DEBUG,
            ):
        """
        Add File handler.
        
        Parameters
        ----------
        logger_path: str
            Path to logger file
        logger_level: int. Defaults to logging.DEBUG = 10
            Create logger object, set formatting and level.
        """
        sh = logging.FileHandler(logger_path)
        sh.setLevel(logger_level)
        sh.setFormatter(self.formatter)
        
        #add handler
        self.logger.addHandler(sh)

def run_simulation(yaml_config_path: str = ""):
    """
    Run a simulation on the linear track.

    Parameters
    ----------
    yaml_config_path: str, Optional (default="")
        Path to the yaml configuration file.
    """ 
    # Load config if argument passed
    if yaml_config_path:
        with open(yaml_config_path,'r') as file:
            conf = yaml.safe_load(file)
            
    # validate yaml cofiguration
    conf = validate_yaml(conf)
    
    # Create logger
    logg = LOGG(logger_level = conf.logger_level)
            
    # Create experiment folder
    date_time = str(datetime.datetime.now()).split('.')[0].replace('-','_').replace(':','_').replace(' ','_')
    newpath = os.path.join(conf.experiment_folder, date_time)
    if conf.store_data:
        os.makedirs(newpath)
        logg.logger.info('Folder {} created!'.format(newpath))
    
    #TODO: add to yaml hyperparams?
    ini_run = 0
    fin_run = 10
    assert fin_run>ini_run, 'should have more than 1 run!'
    tot_runs = fin_run-ini_run
    tot_epochs = 50
    
    #TODO: add various trajectories to yaml? (Linear vs Random?)
    traj_linear_track = np.repeat([np.repeat([[0, 1, 2, 3]], tot_epochs, axis=0)], tot_runs, axis=0)
    traj = traj_linear_track
    T_list = [[100,100,100,100]]*tot_epochs
    
    # TODO: to yaml
    mode = 'MC'
    
    if mode == 'TD':
        fun_to_run = parall_fun
        newpath = 'TD_runs/'+newpath
    elif mode == 'MC':
        fun_to_run = parall_fun_MC
        newpath = 'MC_runs/'+newpath
        
    ### make directory
    os.makedirs(newpath)
    
    if conf.multiprocessing:
        cc = int(.75*multiprocessing.cpu_count())
        pool = Pool(cc)
        output = pool.map(fun_to_run, zip(traj,range(len(traj)),[ini_run]*len(traj),
                                          [newpath]*len(traj), [np.array(T_list)]*len(traj) ) )
        output=np.array(output)
    else:
        output = fun_to_run((traj[0],0,ini_run,
                                newpath, np.array(T_list) ))
        output=np.array([output])
        
        
    outputs = output[:,0]
    
    # pickle.dump(output, open(newpath + '/output_stdp.pkl','wb'))
    #pickle.dump(output_td, open(newpath + '/output_td.pkl','wb'))
    
    l = [np.cumsum([len(x) for x in t]) for t in traj]
    
    params, gamma_var, _, _ = fetch_parameters_new()
    if mode == 'TD':
        w_stdps = np.array([[outputs[i][l_i] for l_i in l[i]] for i in range(len(outputs))])
        w_stdps2 = np.mean(np.sum(w_stdps[:,:,:,:,:,:],axis=-2)/params['N_pre'],axis=-1)
    elif mode == 'MC':
        w_stdps2 = np.array(outputs)
        
    # pickle.dump(w_stdps, open(newpath + '/w_stdps.pkl','wb'))
    # 
    
    pickle.dump(w_stdps2, open(newpath + '/Fig3_w_stdps2.pkl','wb'))
    
    init = np.tile(np.expand_dims(np.expand_dims(np.eye(4),0),0),[tot_runs,1,1,1])
    w_stdps2 = np.concatenate([init,w_stdps2],axis=1)
    
    T_mat = T_random_walk_linear_track(params['no_states']-1)
    M_theor = np.linalg.inv(np.eye(params['no_states']-1)-gamma_var*T_mat[:-1, :-1])
    #
    fig = plt.figure()
    for w1 in range(params['no_states']-1):
        for w2 in range(params['no_states']-1):
            if 3 not in [w1,w2]:
                mn = w_stdps2[:, :, w1, w2].mean(axis=0)
                # std = np.std(w_stdps2[:, :, w1, w2],axis=0)
                plt.plot(mn)
                # plt.fill_between(range(len(std)), mn-std, mn+std,alpha=.3)
                # plt.plot([0,len(w_stdps2[0,:,0,0])],[M_theor[w1, w2],M_theor[w1, w2]])
    # fig.savefig(newpath + '/weights.png')
    
    fig2 = plt.figure()
    root_mse_TD_spiking,std = root_mse(w_stdps2[:,:,:-1,:-1], M_theor, 0)
    plt.plot(root_mse_TD_spiking)    
    plt.fill_between(range(len(std)), root_mse_TD_spiking-std, root_mse_TD_spiking+std,alpha=.3)
    # fig2.savefig(newpath + '/MSE_TD_spiking.png')
    # pickle.dump(root_mse_TD_spiking, open('rmse_TD_id.pkl','wb'))
    
    fs = 16
    
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    
    top = cm.get_cmap('Greys', 128)
    bottom = cm.get_cmap('magma', 128*2)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128*2))[:128]))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    fig3 = plt.figure()
    w_stdp_matrix = plt.imshow(w_stdps2[:,-2,:,:].mean(axis=0),cmap=newcmp,vmin=0,vmax=2) 
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fs)
    plt.yticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
    plt.xticks(ticks=[0,1,2,3],labels=[1,2,3,4],fontsize=fs)
    plt.xlabel('Future state (CA1)',fontsize=fs)
    plt.ylabel('Current state (CA3)',fontsize=fs)
    # fig3.savefig('fig3_.eps',bbox_inches='tight',format='eps')
    
    # import seaborn as sn
    k = np.shape(w_stdps2)[-1]
    from scipy import interpolate
    for i in range(k):
        data = w_stdps2[:,-1,:,i].mean(axis=0)
        plt.figure()
        plt.bar(x = list(range(1,k+1)), height=data,color='b',alpha=.5)#,kde=True,bins=4)
        x = list(range(0,k+2))
        y = np.concatenate([[0],data,[0]])
        f2 = interpolate.interp1d(x, y, kind='quadratic')
        xnew = np.linspace(0,k+1,51)
        plt.fill_between(xnew[:10*(i+2)], 0, f2(xnew)[:10*(i+2)])
        
        # data = [[float(idx)]*int(100*val) for idx,val in enumerate(w_stdps2[:,-1,:,i].mean(axis=0))]
        # sn.kdeplot(x=np.concatenate(data),fill=True,bw_adjust=2.5)
    
    
