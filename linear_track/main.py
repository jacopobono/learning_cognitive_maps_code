"""
Running linear track simulations.

Run from command line using:
    
    `python main.py path_to_yaml`   
"""
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


def learn_SR(traj: np.ndarray) -> np.ndarray :
    """
    Learn the SR using theoretical TD and spiking TD.

    Parameters
    ----------
    traj : np.ndarray
        array containing the trajectory (sequence of states) along
        dimension 1, for each epoch (along dimension 0).

    Returns
    -------
    np.ndarray
        Array containing the final successor matrix learned by spiking TD
        and theoretical TD.

    """
    # Load and set parameters
    params, gamma, eta, lambda_var = parameters_long_linear_track()
    params['trajectories'] = traj
    params['offline'] = False
    
    # calculate the SR using the theoretical TD algorithm
    M_TD = theor_TD_lambda(traj, params['no_states'], gamma, lambda_var, eta)
    
    # calculate the SR using the spiking TD algorithm
    M_stdp = run_spiking_td_lambda(**params)
    return np.array([M_stdp, M_TD], dtype=object)

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
            
    # add file handler to logger
    logger_file = os.path.join(newpath,conf.logger_file) if conf.logger_file else ""
    logg.add_file_handler(logger_path = logger_file)
    
    logg.logger.info('Total runs: {}'.format(conf.tot_runs))
    logg.logger.info('Total epochs: {}'.format(conf.tot_epochs))
    logg.logger.info('Nr of states {}'.format(conf.n_states))
    
    # Create the linear track
    traj_linear_track = np.repeat([np.repeat([np.arange(conf.n_states)], 
                                             conf.tot_epochs, axis=0)], 
                                  conf.tot_runs, axis=0)

    # Run in parallel if more than one run
    if conf.multiprocessing:
        logg.logger.info('Running in parallel...')
        cc = int(conf.cpu_percentage*cpu_count())
        logg.logger.info(f'Using {cc} cpus...')
        pool = Pool(cc)   
        output = pool.map(learn_SR, traj_linear_track)
    else:
        output = learn_SR(traj_linear_track[0])

    # Store output
    if conf.store_data:
        storepath = os.path.join(newpath,"output_stdp.pkl")
        logg.logger.info(f'Storing data in {storepath}')
        pickle.dump(output, open(storepath, 'wb'))

if __name__ == '__main__':
    
    Fire(run_simulation)
    
