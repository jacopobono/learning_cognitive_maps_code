"""
Summary.

This script contains functions to be used when running various 
linear track experiments.
"""
from typing import Tuple
import numpy as np
import tqdm
import logging

logger = logging.getLogger('main_logger')

def create_effective_connections(no_states: int, N_pre: int, N_pre_tot: int, N_post: int) -> np.ndarray:
    """
    For each state, we have N_pre_tot presynaptic neurons and N_post postsyn.
    
    For each postsynaptic neuron, we create N_pre connections out of a 
    possible N_pre_tot.

    Parameters
    ----------
    no_states : int
        Number of states in the environment.
    N_pre : int
        Number of presynaptic neurons per postsynaptic neuron. This is a sub-
        set of N_pre_tot
    N_pre_tot : int
        Total number of presynaptic neurons per state.
    N_post : int
        Total number of postsysnaptic neurons per state.

    Returns
    -------
    effective_connections : np.ndarray
        Array of dimension (no_states,no_states,N_pre_tot,N_post). For each
        possible pre-to-post state, the matrix (N_pre_tot,N_post) defines 
        which neurons are connected (1) and which not (0). Example:
        Connections from state 1 to state 2 are encoded in
        effective_connections[1,2,:,:], where the rows denote the possible
        presynaptic neurons and the columns denote the possible post-
        synaptic neurons. The sum of the columns are equal to N_pre.

    """
    effective_connections = []
    for kk in range(no_states):
        temp1 = []
        for ll in range(no_states):
                temp2 = []
                for mm in range(N_post):
                    temp2.append(np.random.permutation(N_pre*[1] + (N_pre_tot - N_pre)*[0]))
                temp1.append(np.transpose(temp2))
        effective_connections.append(temp1)
    effective_connections = np.array(effective_connections)
    return effective_connections

def convolution(conv: np.ndarray, tau: float, X: np.ndarray, w: float, step: float) -> np.ndarray:
    """
    Convolution operation (exponential window) updating traces.

    Parameters
    ----------
    conv : numpy.ndarray
        array with current values of the trace.
    tau : float
        timeconstant of the convolution window.
    X : numpy.ndarray
        spike times.
    w : float
        weights for the discrete steps.
    step : float
        timestep size.

    Returns
    -------
    conv : numpy.ndarray
        updated array of the traces.

    """
    conv = conv -conv/tau*step + np.multiply(X,w) - conv*(1-step/tau-np.exp(-step/tau))
    return conv

def neuron_model(epsps: np.ndarray, eps0: float, step: float, v0: float, current_state: int) -> np.ndarray:
    """
    Update neuron voltages and generate spikes.

    Parameters
    ----------
    epsps : np.ndarray
        array containing current voltages.
    eps0 : float
        unit epsp amplitude.
    step : float
        timestep size.
    v0 : float
        place-tuned bias input.
    current_state : int
        current state of the agent.

    Returns
    -------
    X : np.ndarray
        array with spikes of the current step.

    """
    u = np.sum(np.sum(epsps*eps0, axis=2), axis=0) #sum over rows (all states) and pre-population
    u[current_state, :] = u[current_state, :] + v0 #place tuned input for active trial
    X=np.random.rand(np.size(u,axis=0), np.size(u,axis=1)) < (u*step)
    return X

def stdp(A_plus: float, 
         A_minus: float, 
         tau_plus: float, 
         tau_minus: float, 
         ca1_spike_train: np.ndarray, 
         ca3_spike_train: np.ndarray, 
         conv_pre: np.ndarray, 
         conv_post: np.ndarray, 
         step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate spike-timing-dependent plasticity.

    Parameters
    ----------
    A_plus : float
        Amplitude of potentiation.
    A_minus : float
        Amplitude of depression.
    tau_plus : float
        Potentiation timescale.
    tau_minus : float
        Depression timescale.
    ca1_spike_train : array
        Array with spikes of CA1 neurons.
    ca3_spike_train : array
        Array with spikes of CA3 neurons.
    conv_pre : array
        Convolution of presynaptic spikes, traces to be used for STDP.
    conv_post : array
        Convolution of postsynaptic spikes, traces to be used for STDP.
    step : float
        Step size.

    Returns
    -------
    W : np.ndarray
        Synaptic weight matrix.
    conv_pre : np.ndarray
        Convolution of presynaptic spikes, traces to be used for STDP..
    conv_post : np.ndarray
        Convolution of postsynaptic spikes, traces to be used for STDP..

    """
    [no_states,no_states,N_pre]=conv_pre.shape
    N_post = np.size(conv_post,axis=2)

    #update trace
    conv_pre = convolution(conv_pre, tau_plus, ca3_spike_train, np.ones([no_states, no_states,N_pre]), step)

    #expand
    ca1_spike_train = np.tile(np.expand_dims(ca1_spike_train,axis=2),[1,1,N_pre,1])
    conv_pre_exp = np.tile(np.expand_dims(conv_pre,axis=3),[1,1,1,N_post])

    #total change in synapse due to stpd
    W = A_plus*conv_pre_exp*ca1_spike_train #+ A_minus*conv_post_exp*ca3_spike_train)
    return (W, conv_pre, conv_post)

def run_spiking_td_lambda(trajectories: list, 
                  effective_connections: np.ndarray, 
                  T: float, 
                  step: float, 
                  no_states: int,
                  N_pre: int, 
                  N_pre_tot: int, 
                  N_post: int, 
                  rate_ca3: float, 
                  eps0: float, 
                  bias: float, 
                  A_plus: float,
                  tau_plus: float, 
                  eta_stdp: float, 
                  A_pre: float, 
                  tau_m: float, 
                  Trials: int, 
                  theta: float, 
                  offline: bool = False, 
                  w_init: np.ndarray = None) -> np.ndarray:
    """
    Run spiking TD lambda.

    Parameters
    ----------
    trajectories : list
        list with the trajectories.
    effective_connections : np.ndarray
        array with effective connections between CA3 and CA1.
    T : float
        Time per state visit.
    step : float
        Timestep.
    no_states : int
        Number of states of the environment.
    N_pre : int
        Number of effective presynaptic neurons per state.
    N_pre_tot : int
        Number of total presynaptic neurons per state.
    N_post : int
        Number of postsynaptic neurons per state.
    rate_ca3 : float
        CA3 firing rate when in the current state.
    eps0 : float
        Unit EPSP.
    bias : float
        Place-tuned bias current.
    A_plus : float
        STDP potentiation amplitude.
    tau_plus : float
        STDP potentiation time constant.
    eta_stdp : float
        STDP learning rate.
    A_pre : float
        STDP presynaptic depression amplitude.
    tau_m : float
        EPSP time constant.
    Trials : int
        Number of trials.
    theta : float
        Duration of the presynaptic activation when in a state.
    offline : bool, optional
        Offline weight updates (after each state visit). The default is False.
    w_init : np.ndarray, optional
        Initial weights. The default is None.

    Returns
    -------
    store_w : np.ndarray
        Stored weight evolution over time.

    """
    # Length of trajectories
    traj_len = [len(t) for t in trajectories]
    # Cumulative length of trajectories
    tot_cumulen = np.cumsum(traj_len)
    # Total number of timesteps
    T_tot = int(T*sum(traj_len)/step)
    # Randomize seed for parallel processing
    np.random.seed()

    # Initialize weight matrix
    w = w_init if w_init else np.tile(np.expand_dims(np.expand_dims(np.identity(no_states), axis=2), axis=3), [1, 1, N_pre, N_post])
    
    # Initialize array to store weights
    store_w = np.zeros((len(traj_len)+1, no_states, no_states, N_pre, N_post))
    store_w[0]=w*effective_connections

    # Initialize spike train arrays, spike traces arrays and epsp array
    ca3_spike_train = np.zeros([1, no_states,N_pre_tot])
    ca1_spike_train = np.zeros([1, no_states,N_post])
    conv_pre=np.zeros([no_states,no_states,N_pre_tot])
    conv_post=np.zeros([no_states,no_states,N_post])
    epsps = np.zeros((no_states, no_states,N_pre_tot,N_post))

    # Initialize trial nr, current_trial and weight change accumulation
    trial_nr = 0
    current_trial = 0
    acc_tot_dw = 0

    # progbar
    progbar = tqdm.tqdm(total=T_tot, desc="STDP Trial {}".format(trial_nr))
    
    # loop over all timesteps
    for i in range(T_tot):

        # calculate current trial and state
        current_trial = current_trial+i//int(T*tot_cumulen[current_trial]/step) #index current trial
        idx_current_state = i//int(T/step) - int(tot_cumulen[current_trial]) # used to find index current state
        curr_state = trajectories[current_trial][idx_current_state] #index current state
        time_state = (i%int(T/step)) #time in current state

        # CA3
        ca3_spike_train = np.zeros([1, no_states, N_pre]) #spike train from CA3 input
        ca3_spike_train[0, curr_state, :] = np.random.rand(N_pre)<(rate_ca3*step)*(time_state<int(theta/step)) #sample Poisson spike train CA3

        # CA1
        epsps = convolution(epsps, tau_m, np.tile(np.expand_dims(np.transpose(ca3_spike_train,axes=[1,0,2]),axis=3),
                                                  [1,no_states,1,N_post]), eps0*w*effective_connections, step) #epsp
        mod_bias = bias*(time_state>= int(theta/step)) # place-tuned bias current
        ca1_spike_train = neuron_model(epsps, 1, step, mod_bias, curr_state) #CA1 spike train

        #weight update
        [dw, conv_pre, conv_post] = stdp(A_plus, 0, tau_plus, 10, np.tile(ca1_spike_train, [no_states, 1, 1]),
                                    np.tile(np.transpose(ca3_spike_train,axes = [1, 0, 2]), [1, no_states, 1]), conv_pre, conv_post, step)
        tot_dw = eta_stdp*dw
        tot_dw[curr_state, :, :, :] = (tot_dw[curr_state, :, :, :] + eta_stdp*A_pre*
                                  w[curr_state, :, :, :]*np.tile(np.expand_dims(ca3_spike_train[0, curr_state, :],axis=1), [1, N_post]))

        if offline == True:
            acc_tot_dw += tot_dw
        elif offline == False:
            np.maximum(w+tot_dw,0,w) #rectify w+tot_dw and store in w
            # w = w + tot_dw
            # w[w<0]=0

        #reset between states
        if int((i+1)%int(T/step)) == 0 and offline:
            np.maximum(w+acc_tot_dw,0,w) #rectify w+tot_dw and store in w 
            # w = w + acc_tot_dw
            # w[w<0]=0
            acc_tot_dw = 0
            
        progbar.update()
        
        #reset between trials
        if ((i+1)%int(tot_cumulen[current_trial]*T/step)) == 0:
            trial_nr += 1
            store_w[trial_nr] = w*effective_connections
            progbar.set_description("STDP Trial {}".format(trial_nr))
            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps
            conv_pre=np.zeros([no_states,no_states,N_pre_tot])
            conv_post=np.zeros([no_states,no_states,N_post])

    progbar.close()
    return store_w



def calculate_parameters_var(
        effective_connections: np.ndarray,
        T: float, 
        step: float, 
        no_states: int,
        N_pre: int, 
        N_pre_tot: int, 
        N_post: int, 
        rate_ca3: float, 
        eps0: float, 
        A_plus: float,
        tau_plus: float, 
        eta_stdp: float, 
        A_pre: float, 
        tau_m: float, 
        Trials: int, 
        theta: float, 
        delay: float) -> Tuple[float,float,float,float]:
    """
    Calcualte TD parameters from STDP parameters.

    Parameters
    ----------
    effective_connections : np.ndarray
        array with effective connections between CA3 and CA1.
    T : float
        Time per state visit.
    step : float
        Timestep.
    no_states : int
        Number of states of the environment.
    N_pre : int
        Number of effective presynaptic neurons per state.
    N_pre_tot : int
        Number of total presynaptic neurons per state.
    N_post : int
        Number of postsynaptic neurons per state.
    rate_ca3 : float
        CA3 firing rate when in the current state.
    eps0 : float
        Unit EPSP.
    A_plus : float
        STDP potentiation amplitude.
    tau_plus : float
        STDP potentiation time constant.
    eta_stdp : float
        STDP learning rate.
    A_pre : float
        STDP presynaptic depression amplitude.
    tau_m : float
        EPSP time constant.
    Trials : int
        Number of trials.
    theta : float
        Duration of the presynaptic activation when in a state.
    delay : float
        Time after theta until next state (delay is T-theta).

    Returns
    -------
    Tuple[float,float,float,float]
        Returns the TD parameters gamma, lambda, eta as well as the bias current.

    """
    # Total time in a state
    T = theta + delay
    
    # Calculation for parameter A
    A1 = N_pre*rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))
    A2 = theta/((tau_m+tau_plus))
    
    # Depression amplitude
    A_LTD = A_pre*rate_ca3*theta
    max_ltd = - A_plus*tau_m*tau_plus*(A1+A2)/theta
    
    # Parameter A
    A = eta_stdp*(eps0*A_plus*rate_ca3*tau_m*tau_plus*(A1+A2) + A_LTD)
    
    # Parameter C
    C = eps0*eta_stdp*A_plus*rate_ca3*tau_plus*(np.exp(theta/tau_plus)-1)*N_pre*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*tau_plus*(1-np.exp(-theta/tau_plus))
    
    # Learning rate
    eta = -A
    
    # Lambda
    lambda_var = 1/(1+C/eta)
    
    # Gamma
    gamma_var = np.exp(-T/tau_plus)/lambda_var

    # Parameter D
    D = eta_stdp*rate_ca3*A_plus*tau_plus*(1-np.exp(-theta/tau_plus))*tau_plus*(1-np.exp(-(T-theta)/tau_plus))
    
    # Bias current
    bias = -A/D
    b = - A_plus*tau_m*tau_plus*(rate_ca3 + 1/(tau_m+tau_plus))
    min_cond = min(max_ltd, b)
    
    
    logger.info('Is A_pre < min? {}'.format( A_pre < min_cond))
    logger.info('Lambda: {}'.format(lambda_var))
    logger.info('Gamma: {}'.format(gamma_var))
    logger.info('eta: {}'.format(eta))
    logger.info('bias: {}'.format(bias))
    logger.info('LTD: {}'.format(A_LTD*eta_stdp))
    logger.info('LTP: {}'.format(eta_stdp*(eps0*A_plus*rate_ca3*tau_m*tau_plus*(A1+A2))))

    return gamma_var, lambda_var, eta, bias

def keep_gamma_eta_same(
        effective_connections: np.ndarray,
        T: float, 
        step: float, 
        no_states: int,
        N_pre: int, 
        N_pre_tot: int, 
        N_post: int, 
        rate_ca3: float, 
        eps0: float, 
        A_plus: float,
        tau_plus: float, 
        eta_stdp: float,
        tau_m: float, 
        Trials: int, 
        theta: float, 
        delay: float, 
        gamma_target: float, 
        eta_target: float) -> Tuple[float,float,float,float,float,float,float]:
    """
    Calcualte STDP parameters from TD parameters eta and gamma.

    Parameters
    ----------
    effective_connections : np.ndarray
        array with effective connections between CA3 and CA1.
    T : float
        Time per state visit.
    step : float
        Timestep.
    no_states : int
        Number of states of the environment.
    N_pre : int
        Number of effective presynaptic neurons per state.
    N_pre_tot : int
        Number of total presynaptic neurons per state.
    N_post : int
        Number of postsynaptic neurons per state.
    rate_ca3 : float
        CA3 firing rate when in the current state.
    eps0 : float
        Unit EPSP.
    A_plus : float
        STDP potentiation amplitude.
    tau_plus : float
        STDP potentiation time constant.
    eta_stdp : float
        STDP learning rate.
    tau_m : float
        EPSP time constant.
    Trials : int
        Number of trials.
    theta : float
        Duration of the presynaptic activation when in a state.
    delay : float
        Time after theta until next state (delay is T-theta).
    gamma_target: float
        Target value for TD parameter gamma (discount)
    eta_target: float
        Target value for TD parameter eta (learning rate)

    Returns
    -------
    Tuple[float,float,float,float]
        Returns the TD parameters gamma, lambda, eta as well as the bias current.

    """
    A1 = N_pre*rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))
    A2 = theta/((tau_m+tau_plus))
    LTP = eta_stdp*eps0*(A_plus*rate_ca3*tau_m*tau_plus*(A1+A2))
    # Costraint 1
    A_pre =   -(eta_target + LTP)/(eta_stdp*rate_ca3*theta)
    A_LTD = eta_stdp*A_pre*rate_ca3*theta
    max_ltd = - eps0*A_plus*tau_m*tau_plus*(A1+A2)/theta
    A = LTP + A_LTD
    C = eps0*eta_stdp*A_plus*rate_ca3*tau_plus*(np.exp(theta/tau_plus)-1)*N_pre*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*tau_plus*(1-np.exp(-theta/tau_plus))
    # Costraint 2
    const = (1-C/A)*np.exp(-theta/tau_plus)
    delay = tau_plus*(-np.log(gamma_target) + np.log(const))
    T = theta + delay

    gamma_var, lambda_var, eta, bias = calculate_parameters_var(effective_connections,T, step, no_states,
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, A_plus,
                   tau_plus, eta_stdp, A_pre, tau_m, Trials, theta, delay)
#    eta = -A
#    lambda_var = 1/(1+C/eta)
#    gamma_var = np.exp(-T/tau_plus)/lambda_var
#
    D = eta_stdp*rate_ca3*A_plus*tau_plus*(1-np.exp(-theta/tau_plus))*tau_plus*(1-np.exp(-(T-theta)/tau_plus))*bias
#    gamma_var = np.exp(-T/tau_plus)/lambda_var
#    bias = -A/D
    LTP = eta_stdp*eps0*(A_plus*rate_ca3*tau_m*tau_plus*(A1+A2))
    b = - A_plus*tau_m*tau_plus*(rate_ca3 + 1/(tau_m+tau_plus))
    A_LTD = eta_stdp*A_pre*rate_ca3*theta
    min_cond = min(max_ltd, b)
    
    logger.info('Is A_pre < min? ' + str( A_pre < min_cond))
    logger.info('Lambda: '+str(lambda_var))
    logger.info('Gamma: '+str(gamma_var))
    logger.info('eta: '+str(eta))
    logger.info('bias: '+str(bias))
    logger.info('A_pre: '+str(A_pre))
    logger.info('delay: '+str(delay))
    logger.info('T: '+str(T))
    logger.info('LTP: '+str(LTP))
    logger.info('LTP2: '+str(D))
    logger.info('LTD: '+str(A_LTD))
    logger.info('B/A: '+str(D/A))

    return gamma_var, lambda_var, eta, A_pre, delay, T, bias

def parameters_long_linear_track() -> Tuple[dict,float,float,float]:
    """
    Generate parameters for the linear track.

    Returns
    -------
    Tuple[dict,float,float,float]
        Return a parameters dictionary, and the parameters for the 
        theoretical TD lambda.

    """
    params = {
           'A_plus': 1,
           'T': 100,
           'eta_stdp': 0.002,
           'no_states': 10,
           'rate_ca3': 0.1,
           'step': 0.1,
           'tau_m': 2,
           'tau_plus': 20,
           'N_post':1,
           'N_pre_tot':20,
           'N_pre':20,
           'Trials': 400,
           'theta': 200
           }
    params['eps0'] = 1/params['N_pre']
    params['delay'] = params['T'] - params['theta']
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])

    params['gamma_target'] = 0.5
    params['eta_target'] = 0.01
    gamma, lam, eta, params['A_pre'], params['delay'], params['T'], params['bias'] = keep_gamma_eta_same(**params)
    del params['delay']
    del params['gamma_target']
    del params['eta_target']
    return params, gamma, eta, lam


def TD_lambda_update(
        curr_state: int,
        next_state: int,
        gamma: float,
        lam: float,
        alpha: float,
        elig_trace: np.ndarray,
        M: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    TD lambda update.

    Parameters
    ----------
    curr_state : int
        current state.
    next_state : int
        next state.
    gamma : float
        delay discount parameter.
    lam : float
        TD lambda parameter.
    alpha : float
        learning rate.
    elig_trace : np.ndarray
        eligibility trace.
    M : np.ndarray
        successor matrix.

    Returns
    -------
    M : np.ndarray
        updated successor matrix.
    elig_trace : np.ndarray
        updated eligibility traces.

    """
    n_states = np.shape(M)[0]
    one_vector = np.zeros(n_states)
    one_vector[curr_state] = 1
    elig_trace = lam*gamma*elig_trace + 1*one_vector
    pred_error = one_vector + gamma*M[next_state] - M[curr_state]
    M = M + alpha*np.outer(elig_trace, pred_error)
    return M, elig_trace

def theor_trajectory(trajectory: list, 
                    n_states: int, 
                    gamma: float, 
                    lam: float, 
                    alpha: float, 
                    M=None) -> np.ndarray:
    """
    Run theoretical TD lambda on one trajectory.

    Parameters
    ----------
    trajectory : list
        trajectory to be run.
    n_states : int
        number of states in the environment.
    gamma : float
        discount parameter.
    lam : float
        lambda parameter regulating TD(lambda).
    alpha : float
        learning rate.
    M : TYPE, optional
        initial successor matrix. The default is None.

    Returns
    -------
    M : np.ndarray
        final successor matrix.

    """
    # Initialize SR matrix and SR matrix storing var
    if M is None:
        M = np.eye(n_states)
    # Initialize eligibility trace
    elig_trace = np.zeros(n_states)

    # loop over states
    for curr_state, next_state in zip(trajectory, trajectory[1:]):
        M, elig_trace = TD_lambda_update(
                            curr_state,
                            next_state,
                            gamma,
                            lam,
                            alpha,
                            elig_trace,
                            M
                            )
        
    return M

def theor_TD_lambda(trajectories: list, 
                    n_states: int, 
                    gamma: float, 
                    lam: float, 
                    alpha: float, 
                    M=None) -> np.ndarray:
    """
    Run theoretical TD lambda for multiple trajectories.

    Parameters
    ----------
    trajectories : list
        list of trajectories.
    n_states : int
        number of states in the environment.
    gamma : float
        discount parameter.
    lam : float
        lambda parameter regulating TD(lambda).
    alpha : float
        learning rate.
    M : TYPE, optional
        initial successor matrix. The default is None.

    Returns
    -------
    store_M : np.ndarray
        final successor matrix after each trial.

    """
    # Initialize SR matrix and SR matrix storing var
    if M is None:
        M = np.eye(n_states)
    n_trials = len(trajectories)
    tot_len = sum([len(t) for t in trajectories])
    store_M = np.zeros((n_trials, n_states, n_states))
    store_M[0] = M

    # progress bar
    progbar = tqdm.tqdm(total=tot_len, desc="TD Trial {}".format(0))
    
    # Loop over trials
    for trial in range(n_trials):
        # Get current trajectory
        curr_traj = trajectories[trial]
        
        M = theor_trajectory(curr_traj, 
                    n_states, 
                    gamma, 
                    lam, 
                    alpha, 
                    M)
            
        # Store M at the end of each trial
        store_M[trial] = M
        progbar.set_description( "TD Trial {}".format(trial+1))
    progbar.close()
    return store_M
