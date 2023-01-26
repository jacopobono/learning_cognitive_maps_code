"""
Summary.

This script contains functions to be used when running various
linear track experiments.
"""
from typing import Tuple
import numpy as np
import tqdm

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

def neuron_model(epsps: np.ndarray, eps0: float,
                 step: float, v0: float,
                 current_state: int, distances: np.ndarray = np.array([])) -> np.ndarray:
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
    distances : np array
        distances of all states to current location.

    Returns
    -------
    X : np.ndarray
        array with spikes of the current step.

    """

    u = np.sum(np.sum(epsps*eps0, axis=2), axis=0) #sum over rows (all states) and pre-population. Final dims = [n_states, N_post] or [1m1] for value neuron
    if len(distances) == 0:
        if v0>0:
            u[current_state, :] = u[current_state, :] + v0 #place tuned input for active trial
    else:
        u[:, :] = u[:, :] + v0*np.expand_dims(distances, axis=1) #*np.tile(distances, [np.size(u,axis=1),1])
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
        tau_m: float,
        theta: float,
        delay: float,
        verbose: bool = False,
        A_pre = None,
        ) -> Tuple[float,float,float,float]:
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
    tau_m : float
        EPSP time constant.
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
    # A1 = 1*rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))
    A2 = theta/((tau_m+tau_plus))

    # Depression amplitude
    max_ltd = - A_plus*tau_m*tau_plus*(A1+A2)/theta

    if not A_pre:
        A_pre = max_ltd - 5

    A_LTD = A_pre*rate_ca3*theta

    # Parameter A
    A = eta_stdp*(eps0*A_plus*rate_ca3*tau_m*tau_plus*(A1+A2) + A_LTD)
    # A = eta_stdp*(1*A_plus*rate_ca3*tau_m*tau_plus*(A1+A2) + A_LTD)

    # Parameter C
    C = eps0*eta_stdp*A_plus*rate_ca3*tau_plus*(np.exp(theta/tau_plus)-1)*N_pre*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*tau_plus*(1-np.exp(-theta/tau_plus))
    # C = 1*eta_stdp*A_plus*rate_ca3*tau_plus*(np.exp(theta/tau_plus)-1)*1*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*tau_plus*(1-np.exp(-theta/tau_plus))

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

    if verbose:
        print('Is A_pre < min? {}'.format( A_pre < max_ltd))
        print('A_pre: {}'.format(A_pre))
        print('max_ltd: {}'.format(max_ltd))
        print('Lambda: {}'.format(lambda_var))
        print('Gamma: {}'.format(gamma_var))
        print('eta: {}'.format(eta))
        print('bias: {}'.format(bias))
        print('LTD: {}'.format(A_LTD*eta_stdp))
        print('LTP: {}'.format(eta_stdp*(eps0*A_plus*rate_ca3*tau_m*tau_plus*(A1+A2))))

    return gamma_var, lambda_var, eta, bias, A_pre

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

    gamma_var, lambda_var, eta, bias, _ = calculate_parameters_var(effective_connections,T, step, no_states,
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, A_plus,
                   tau_plus, eta_stdp, tau_m, theta, delay)
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

    return gamma_var, lambda_var, eta, A_pre, delay, T, bias


def calculate_parameters_TD_new(
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
        ) -> Tuple[float,float,float,float,float,float,float]:
    """
    Calcualte STDP parameters.

    Parameters
    ----------
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
    A1 = rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))*N_pre
    A2 = theta/((tau_m+tau_plus))
    max_ltd = A_plus*tau_m*tau_plus*(A1+A2)/theta

    A_pre = -max_ltd-5  # subtract by 5 to ensure positive learning rate

    # Calculate A
    A_LTD = A_pre*rate_ca3*theta
    A = eta_stdp*eps0*(A_plus*rate_ca3*tau_m*tau_plus*(A1+A2) + A_LTD)

    # Calculate D
    D = eta_stdp*rate_ca3*A_plus*tau_plus*(1-np.exp(-theta/tau_plus))*tau_plus*(1-np.exp(-(T-theta)/tau_plus))

    # Calculate C
    C = eta_stdp*A_plus*rate_ca3*tau_plus*(np.exp(theta/tau_plus)-1)*N_pre*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*tau_plus*(1-np.exp(-theta/tau_plus))

    # Calculate parameters TD
    eta_var = -A
    lambda_var = 1/(1+C/eta_var)
    gamma_var = np.exp(-T/tau_plus)/lambda_var

    # Calculate background firing rate
    bias = -A/D

    return eta_var, lambda_var, gamma_var, bias, A_pre

def parameters_linear_track(no_states: int) -> Tuple[dict,float,float,float]:
    """
    Generate parameters for the linear track.

    Parameters
    ----------
    no_states: int
        Number of states in the linear track.

    Returns
    -------
    Tuple[dict,float,float,float]
        Return a parameters dictionary, and the parameters for the
        theoretical TD lambda.

    """
    params = {
           'A_plus': 1,
           'T': 100,
           'eta_stdp': 0.003, #0.002
           'no_states': no_states,
           'rate_ca3': 0.1,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60, #20
           'N_post':1,
           'N_pre_tot':1,
           'N_pre':1,
           'theta': 80, #200
           }
    params['eps0'] = 1/params['N_pre']
    params['delay'] = params['T'] - params['theta']
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])

    gamma, lam, eta, params["bias"], params["A_pre"] = calculate_parameters_var(**params)



    del params['delay']
    return params, gamma, eta, lam

def parameters_linear_track_value(no_states: int, T: int, alpha: float, A_pre: int) -> Tuple[dict,float,float,float]:
    """
    Generate parameters for the linear track.

    Parameters
    ----------
    no_states: int
        Number of states in the linear track.

    Returns
    -------
    Tuple[dict,float,float,float]
        Return a parameters dictionary, and the parameters for the
        theoretical TD lambda.

    """
    params = {
           'A_plus': 1,
           'T': T,
           'eta_stdp': 0.003,
           'no_states': no_states,
           'rate_ca3': 0.1,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60,
           'N_post':250, #250
           'N_pre_tot':1000, #1000
           'N_pre':250, #250
           'theta': alpha*T,
           }
    params['eps0'] = 1/params['N_pre']
    params['delay'] = (1-alpha)*params['T']
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])

    gamma, lam, eta, params["bias"], params["A_pre"] = calculate_parameters_var(**params, A_pre = A_pre)


    del params['delay']
    return params, gamma, eta, lam

def parameters_linear_track_population(rate, tau_plus, tau_m, no_states: int) -> Tuple[dict,float,float,float]:
    """
    Generate parameters for the linear track.

    Parameters
    ----------
    no_states: int
        Number of states in the linear track.

    Returns
    -------
    Tuple[dict,float,float,float]
        Return a parameters dictionary, and the parameters for the
        theoretical TD lambda.

    """
    params = {
           'A_plus': 1,
           'T': 100,
           'eta_stdp': 0.001,
           'no_states': no_states,
           'rate_ca3': rate, #0.1,
           'step': 0.01,
           'tau_m': tau_m, #2,
           'tau_plus': tau_plus, #200,
           'N_post':10, #250
           'N_pre_tot':50, #1000
           'N_pre':10, #250
           'theta': 80,
           }
    params['eps0'] = 1/params['N_pre']
    params['delay'] = params['T'] - params['theta']
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])

    gamma, lam, eta, params["bias"], params["A_pre"] = calculate_parameters_var(**params)

    del params['delay']
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
    for curr_state, next_state in zip(trajectory[:-1], trajectory[1:]):
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
    store_M = np.zeros((n_trials, n_states, n_states))
    store_M[0] = M

    # progress bar
    progbar = tqdm.tqdm(total=n_trials, desc="TD Trial {}".format(0))

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




def run_MC(trajectories, temporal_between,temporal_same, effective_connections, T, step, no_states, N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus,
                   tau_plus, eta_stdp, A_pre, tau_m,
                  pre_offset, theta, w = [], ini='identity',spike_noise_prob=0.2):#(trajectories,w,temporal_between,temporal_same, tau_m, Trials, pre_offset, A_pre, T, no_states, bias,
                   # A_minus,tau_minus,A_plus,tau_plus,eta_stdp,step,N_pre_tot,N_post,N_pre,eps0,store_w,store_w_end):

    #no_states = np.size(w,axis=0)
    np.random.seed() #randomize seed for parallel

    T = temporal_between #time in each state
    same_loc = temporal_same
    A_pre = -A_plus*np.exp(-(same_loc)/tau_plus)
    T_tot = int(T/step)

    if w == []:
        init = np.identity(no_states) if ini=='identity' else 2*np.random.rand(no_states,no_states)
        w_init = np.tile(np.expand_dims(np.expand_dims(init, axis=2), axis=3), [1, 1, N_pre_tot, N_post])
        w = w_init*effective_connections
    store_w = []
    store_w_end = []

#    for traj in trajectories:

    #noise_steps = 4
    interval_range = 50 # in units of steps, default step is 0.01 ms

    assert interval_range < int(same_loc/step), 'wrong interval_range?'

    wavg = []

    print('Calculating TD(1) spiking ....')

    # For loop over epochs in trajectory
    for epoch in trajectories:

        trial_len = len(epoch)

        conv_pre=np.zeros([no_states,no_states,N_pre_tot])
        conv_post=np.zeros([no_states,no_states,N_post])

        w_offline = 0


#        noise1 = - np.random.randint(noise_steps)
#        noise2 = 0
        # For loop over states in trial
        for j in range(trial_len):
            curr_state = epoch[j] #index current state

            p = spike_noise_prob

            #choose with probability p two pre spikes, otherwise 1
            n_pre_spikes = np.random.binomial(1, p)+1
            # if 2 spikes, choose with 50% probability to use two spikes, otherwise 0
            if n_pre_spikes == 2:
                n_pre_spikes = np.random.binomial(1, 0.5)*2


            #choose with probability p two post spikes
            n_post_spikes =np.random.binomial(1, p)+1
            # if 2 spikes, choose with 50% probability to use two spikes, otherwise 0
            if n_post_spikes == 2:
                n_post_spikes = np.random.binomial(1, 0.5)*2

            times_pre = np.random.choice(np.arange(0, interval_range), n_pre_spikes)
            times_post = np.random.choice(np.arange(int(same_loc/step), interval_range + int(same_loc/step)), n_post_spikes)


            for i in np.arange(T_tot):

                ca3_spike_train = np.zeros([1, no_states,N_pre_tot])
                ca1_spike_train = np.zeros([1, no_states,N_post])


                if i in times_pre:#(i+noise1)%(T/step)==0: #or (i+1)%(T/step)==0 or (i+2)%(T/step)==0:
                    spike_ca3 = 1#np.random.rand() < 1
                    spike_ca1 = 0
                    #noise2 = - np.random.randint(noise_steps)

                elif i in times_post:#(((i+noise2)-int(same_loc/step))%(T/step)==0): #or (((i+1)-int(same_loc/step))%(T/step)==0) or (((i+2)-int(same_loc/step))%(T/step)==0):
                    spike_ca1 = 1#np.random.rand() < 1
                    spike_ca3 = 0
                    #noise1 = - np.random.randint(noise_steps)
            #
                else:
                    spike_ca3 = 0
                    spike_ca1 = 0

                ca3_spike_train[0,curr_state,:]= spike_ca3#(i%(T/step)==0)*1
                ca1_spike_train[0,curr_state,:] = spike_ca1#((i-int(same_loc/step))%(T/step)==0)*1

                #weight update
                [dw, conv_pre, conv_post]=stdp(A_plus, 0, tau_plus, 10, np.tile(ca1_spike_train,[no_states,1,1]),
                                        np.tile(np.transpose(ca3_spike_train,axes=[1,0,2]),[1,no_states,1]), conv_pre, conv_post, step)
                tot_dw = eta_stdp*dw

                tot_dw[curr_state, :, :, :] = (tot_dw[curr_state, :, :, :] + eta_stdp*A_pre*
                                          w[curr_state, :, :, :]*np.tile(np.expand_dims(ca3_spike_train[0, curr_state, :],axis=1), [1, N_post]))


                w_offline = w_offline + tot_dw
                w = w + tot_dw
                w[w<0]=0

#                if spike_ca1==1:
#                    break
#            w = w + w_offline
#            w[w<0]=0

        wavg.append(np.mean(np.sum(w[:,:,:,:],axis=2)/N_pre,axis=2))
        store_w.append(np.mean(wavg,axis=0))
        store_w_end.append(np.mean(np.sum(w[:,:,:,:],axis=2)/N_pre,axis=2))

    return store_w_end




def run_td_lambda_new_continuousTime(trajectories, effective_connections, T_lists, step, no_states,
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus,
                   tau_plus, eta_stdp, A_pre, tau_m,
                  theta, offline=False,w=[],ini='identity',state_3_rate=1):

    traj_len = [len(t) for t in T_lists]
    tot_cumulen = np.cumsum(traj_len)
    Trials = len(T_lists)
    T_tot = int(sum([sum(x) for x in T_lists])/step)

    np.random.seed() #randomize seed for parallel

    #initialize weight matrix
    if w == []:
        init = np.identity(no_states) if ini=='identity' else 2*np.random.rand(no_states,no_states)
        w_init = np.tile(np.expand_dims(np.expand_dims(init, axis=2), axis=3), [1, 1, N_pre_tot, N_post])
        w = w_init

    store_w = np.zeros((sum(traj_len)+1, no_states, no_states, N_pre_tot, N_post))
    store_w[0]=w*effective_connections

    # initialize
    ca3_spike_train = np.zeros([1, no_states,N_pre_tot]) #spike train from CA3 input
    ca1_spike_train = np.zeros([1, no_states,N_post])
    conv_pre=np.zeros([no_states,no_states,N_pre_tot])
    conv_post=np.zeros([no_states,no_states,N_post])
    epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps

    trial_nr = 0
    print('Calculating TD lambda spiking ....')

    current_trial = 0
    acc_tot_dw = 0
    idx_current_state = 0

    pot = np.empty_like(w)*0
    dep = np.empty_like(w)*0

    cumul_trials = 0
    Trial_tot_t = 0
    prev_trial = 0
    prev_state = 0
    state_counter = 0

    for i in range(T_tot):

        T = T_lists[current_trial][idx_current_state]
        Ts_trial = [t/step for t in T_lists[current_trial]]
        Trial_tot_t = sum(Ts_trial)

        # calculate current trial and state
        states_trial = [y for y,x in enumerate(T_lists[current_trial]) for _ in range(int(x/step)) ]
        assert Trial_tot_t == len(states_trial), f'somthing wrong {Trial_tot_t} vs {len(states_trial)}'

        prev_trial = current_trial
        current_trial = prev_trial+int((i)//(cumul_trials+int(Trial_tot_t))) #index current trial

        if prev_trial != current_trial:
            cumul_trials += Trial_tot_t

        t_trial = int(i - cumul_trials)

        prev_state = idx_current_state
        idx_current_state = states_trial[t_trial]

        # print(f'idx_current_state {idx_current_state}')
        curr_state = trajectories[current_trial][idx_current_state] #index current state
        t_start_state = states_trial.index(idx_current_state)

        time_state = t_trial - t_start_state #time in current state

        # CA3
        ca3_spike_train = np.zeros([1, no_states, N_pre_tot]) #spike train from CA3 input

        if idx_current_state == 2:
            rate_ca3_s = rate_ca3*state_3_rate
        else:
            rate_ca3_s = rate_ca3

        ca3_spike_train[0, curr_state, :] = np.random.rand(N_pre_tot)<(rate_ca3_s*step)*(time_state<int(theta/step)) #sample Poisson spike train CA3

        # CA1
        epsps = convolution(epsps, tau_m, np.tile(np.expand_dims(np.transpose(ca3_spike_train,axes=[1,0,2]),axis=3),
                                                  [1,no_states,1,N_post]), w*effective_connections, step) #epsp
        mod_bias = bias*(time_state>= int(theta/step))
        #store_bias[i] = (mod_bias)
        if time_state == int(theta/step):
            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps

        ca1_spike_train = neuron_model(epsps, eps0/N_pre, step, mod_bias, curr_state) #CA1 spike train

        #weight update
        [dw, conv_pre, conv_post] = stdp(A_plus, 0, tau_plus, 10, np.tile(ca1_spike_train, [no_states, 1, 1]),
                                    np.tile(np.transpose(ca3_spike_train,axes = [1, 0, 2]), [1, no_states, 1]), conv_pre, conv_post, step)
        tot_dw = eta_stdp*dw
        pot += eta_stdp*dw
        dep[curr_state, :, :, :] += eta_stdp*eps0*A_pre* w[curr_state, :, :, :]*np.tile(np.expand_dims(ca3_spike_train[0, curr_state, :],axis=1), [1, N_post])
        tot_dw[curr_state, :, :, :] = (tot_dw[curr_state, :, :, :] + eta_stdp*eps0*A_pre*
                                  w[curr_state, :, :, :]*np.tile(np.expand_dims(ca3_spike_train[0, curr_state, :],axis=1), [1, N_post]))

        if offline == True:
            acc_tot_dw += tot_dw
        elif offline == False:
            w = w + tot_dw#*np.abs(w_init-1)
            w[w<0]=0

        #reset between states
        if prev_state != idx_current_state:

            if offline == True:
                w = w + acc_tot_dw #*np.abs(w_init-1)
                w[w<0]=0
                acc_tot_dw = 0

            store_w[state_counter, :, :, :, :] = w*effective_connections

            state_counter += 1
            pot = np.empty_like(w)*0
            dep = np.empty_like(w)*0
        #reset between trials
        if prev_trial != current_trial:


            trial_nr += 1

            print(f'trial nr: {trial_nr}')


            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps
            conv_pre=np.zeros([no_states,no_states,N_pre_tot])
            conv_post=np.zeros([no_states,no_states,N_post])

    store_w[state_counter+1, :, :, :, :] = w*effective_connections

    return store_w

def run_value_function(trajectories, effective_connections, T_lists, step, no_states,
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus,
                   tau_plus, eta_stdp, A_pre, tau_m,
                  theta, wini, reward_vec, offline=False,):


        assert no_states == len(reward_vec), 'ERROR: reward_vec should have length equal to no_states'
        assert np.shape(wini) == (no_states, no_states, N_pre_tot, N_post), 'ERROR: initial weights wini should have shape (no_states, no_states, N_pre_tot, N_post)'

        traj_len = [len(t) for t in T_lists]
        tot_cumulen = np.cumsum(traj_len)
        Trials = len(T_lists)
        T_tot = int(sum([sum(x) for x in T_lists])/step)

        np.random.seed() #randomize seed for parallel

        #initialize weight matrix
        w = wini

        # initialize
        ca3_spike_train = np.zeros([1, no_states,N_pre_tot]) #spike train from CA3 input
        ca1_spike_train = np.zeros([1, no_states,N_post])
        epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps

        nr_value_neurons = 250
        v_spike_train = np.zeros([nr_value_neurons])
        epsps_v = np.zeros((no_states,1,N_post,nr_value_neurons))

        trial_nr = 0
        print('Calculating TD lambda spiking ....')

        current_trial = 0
        idx_current_state = 0

        cumul_trials = 0
        Trial_tot_t = 0
        prev_trial = 0
        prev_state = 0
        state_counter = 0

        reward_neuron = np.zeros((no_states, nr_value_neurons, T_tot))

        for i in tqdm.tqdm(range(T_tot)):

            T = T_lists[current_trial][idx_current_state]
            Ts_trial = [t/step for t in T_lists[current_trial]]
            Trial_tot_t = sum(Ts_trial)

            # calculate current trial and state
            states_trial = [y for y,x in enumerate(T_lists[current_trial]) for _ in range(int(x/step)) ]
            assert Trial_tot_t == len(states_trial), f'somthing wrong {Trial_tot_t} vs {len(states_trial)}'

            prev_trial = current_trial
            current_trial = prev_trial+int((i)//(cumul_trials+int(Trial_tot_t))) #index current trial

            if prev_trial != current_trial:
                cumul_trials += Trial_tot_t

            t_trial = int(i - cumul_trials)

            prev_state = idx_current_state
            idx_current_state = states_trial[t_trial]

            # print(f'idx_current_state {idx_current_state}')
            curr_state = trajectories[current_trial][idx_current_state] #index current state
            t_start_state = states_trial.index(idx_current_state)

            time_state = t_trial - t_start_state #time in current state

            # CA3
            ca3_spike_train = np.zeros([1, no_states, N_pre_tot]) #spike train from CA3 input

            if idx_current_state == 2:
                rate_ca3_s = rate_ca3#*0.5
            else:
                rate_ca3_s = rate_ca3

            ca3_spike_train[0, curr_state, :] = np.random.rand(N_pre_tot)<(rate_ca3_s*step)*(time_state<int(theta/step)) #sample Poisson spike train CA3

            # CA1
            # transpose to dims [n_states, 1, N_pre_tot, 1]
            # tile to dims [n_states, n_states, N_pre_tot, N_post]
            # element-wise multiplication with weights w
            epsps = convolution(epsps, tau_m, np.tile(np.expand_dims(np.transpose(ca3_spike_train,axes=[1,0,2]),axis=3),
                                                      [1,no_states,1,N_post]), w*effective_connections, step) #epsp


            mod_bias = bias*(time_state>= int(theta/step))

            ca1_spike_train = neuron_model(epsps, eps0, step, mod_bias, curr_state) #CA1 spike train

            # VALUE
            # ca1_spike_train has dim [n_states, n_post] and is reshapen to dims [n_states, 1, n_post, 1]
            # reward_vec has dims [n_states] and is reshapen to [n_states, 1, n_post, nr_value_neurons]
            #
            #
            rew_weights = np.tile(np.expand_dims(np.expand_dims(np.expand_dims(reward_vec,1),2),3), [1,1,N_post,nr_value_neurons])
            epsps_v = convolution(epsps_v, tau_m, np.expand_dims(np.expand_dims(ca1_spike_train,axis=1),axis=3), rew_weights, step) #epsp
            value_spike_train = neuron_model(epsps_v, eps0*N_pre/N_post, step, 0, curr_state) #VALUE spike train
            if curr_state != prev_state:
                print(curr_state)
            reward_neuron[curr_state, :, i] = value_spike_train

        return reward_neuron

