import numpy as np 
from utils_nn import create_effective_connections

def calculate_parameters_TD(T, A_plus, tau_plus, rate_ca3, eta_stdp, tau_m, N_pre, eps0, pre_offset, no_states, step, N_post, N_pre_tot, Trials, delay=0, A_pre=0):
    
    # I think this has to do with scaling of LTP with number of neurons - ASK JACOPO 
    max_ltd = A_plus*tau_m*(N_pre*rate_ca3*tau_plus
            + tau_plus/(tau_m+tau_plus))
    A_pre = -max_ltd-pre_offset#*.1
    
    # Calculate A
    A1 = rate_ca3*tau_plus*(1-np.exp(-T/tau_m))*(T-tau_plus*(1-np.exp(-T/tau_plus)))*N_pre
    A2 = T*tau_plus/(tau_m+tau_plus)
    A_LTD = A_pre*rate_ca3*T
    A = eta_stdp*eps0*(A_plus*rate_ca3*tau_m*(A1+A2) + A_LTD)

    print('=====')
    print('A:'+str(A))
    #A_corr = np.exp(A) -1
    #print('Acorr:'+str(A_corr))
    #A = A_corr
    print('=====')
    
    print('=====')
    print('A_LTD:'+str(A_LTD))
    #A_corr = np.exp(A) -1
    #print('Acorr:'+str(A_corr))
    #A = A_corr
    print('=====')
    # Calculate B
    B = eta_stdp*A_plus*(T-tau_plus*(1-np.exp(-T/tau_plus)))*rate_ca3*tau_plus
    B_delayed = eta_stdp*A_plus*(T-delay-tau_plus*(np.exp(-delay/tau_plus)-np.exp(-T/tau_plus)))*rate_ca3*tau_plus
    print('B:'+str(B))
    print('B:'+str(B_delayed))
    print('A/B_delayed: '+str(-A/B_delayed))
    print('=====')

    # Calculate C
    C0 = N_pre*eps0*rate_ca3*tau_m*(1-np.exp(-T/tau_m))*(tau_plus*(1-np.exp(-T/tau_plus)))
    C1 = A_plus*tau_plus*rate_ca3*(np.exp(T/tau_plus)-1)
    C = eta_stdp*C0*C1
    print('C:'+str(C))
#    C_corr = np.exp(C*np.exp(-T/tau_plus))-1
#    print('Ccorr:'+str(C_corr))
    print('=====')
    
    # Calculate D
    D = C*tau_plus*(1-np.exp(-T/tau_plus))/C0
    D_delayed = C*tau_plus*(np.exp(-delay/tau_plus)-np.exp(-T/tau_plus))/C0
    print('D: '+str(D))
    print('D delayed: '+str(D_delayed))
    print('D_delayed/A: '+str(-D_delayed/A))
    print('A/D_delayed: '+str(-A/D_delayed))

#    C_corr = np.exp(C*np.exp(-T/tau_plus))-1
#    print('Ccorr:'+str(C_corr))
    print('=====')

    # Calculate parameters TD
    eta = -A
    lambda_var = 1/(1+C/eta)
    gamma_var = np.exp(-T/tau_plus)/lambda_var

    # Calculate spontaneous rate
    bias_TD0 = -A/B #spontaneous rate
    bias_TDlambda = -A/D  
    bias_avg = .50*bias_TD0 
    
    print('A pre: '+str(A_pre))
    print('Bias TD(0): '+str(bias_TD0))
    print('Bias TD(' +  chr(955) + '): ' + str(bias_TDlambda))
    print('Bias average: '+str(bias_avg))
    
    print('=====')
    print('Lambda: '+str(lambda_var))
    print('Gamma: '+str(gamma_var))
    print('eta: '+str(eta))

    return eta, lambda_var, gamma_var, bias_TD0, bias_TDlambda, bias_avg, A_pre, B

def calculate_parameters_TD_new(T, A_plus, tau_plus, rate_ca3, eta_stdp, tau_m, N_pre, eps0, pre_offset, no_states, step, N_post, N_pre_tot, Trials, theta, A_pre=0, verbose=False):
    
    # I think this has to do with scaling of LTP with number of neurons - ASK JACOPO 
    #max_ltd = A_plus*tau_m*(1/(1-np.exp(-1/(rate_ca3*tau_plus)))+ tau_plus/(tau_m+tau_plus))
    A1 = rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))*N_pre
    A2 = theta/((tau_m+tau_plus))
    max_ltd = A_plus*tau_m*tau_plus*(A1+A2)/theta
#    max_ltd = A_plus*tau_m*(N_pre*rate_ca3*tau_plus
#            + tau_plus/(tau_m+tau_plus))
    A_pre = -max_ltd-pre_offset 

    # Calculate A
    A_LTD = A_pre*rate_ca3*theta
    A = eta_stdp*eps0*(A_plus*rate_ca3*tau_m*tau_plus*(A1+A2) + A_LTD)
    

    D = eta_stdp*rate_ca3*A_plus*tau_plus*(1-np.exp(-theta/tau_plus))*tau_plus*(1-np.exp(-(T-theta)/tau_plus))

    # Calculate C
    C = eta_stdp*A_plus*rate_ca3*tau_plus*(np.exp(theta/tau_plus)-1)*N_pre*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*tau_plus*(1-np.exp(-theta/tau_plus))
#    C = D*N_pre*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*np.exp(theta/tau_plus)
    
    # Calculate parameters TD
    eta = -A
    lambda_var = 1/(1+C/eta)
    gamma_var = np.exp(-T/tau_plus)/lambda_var

    # Calculate spontaneous rate
    bias = -A/D  
    
    if verbose == True: 
        print('A1:' + str(A1))
        print('A2:' + str(A2))
        print('max_ltd:' + str(max_ltd))
    
        print('=====')
        print('A:'+str(A))
    #    A_corr = np.exp(A) -1
    #    print('Acorr:'+str(A_corr))
        print('=====')
        
        print('=====')
        print('A_LTD:'+str(A_LTD))
        print('D: '+str(D))
        print('C:'+str(C))
        print('=====')
        print('A pre: '+str(A_pre))
        print('Bias: '+str(bias))
    
        print('=====')
        print('Lambda: '+str(lambda_var))
        print('Gamma: '+str(gamma_var))
        print('eta: '+str(eta))


    return eta, lambda_var, gamma_var, bias, A_pre






def theor_TD_lambda(traj, gamma, lambda_par, nr_states, alpha, M_TD_lambda=[]):
    
    if M_TD_lambda == []:
        M_TD_lambda = np.eye(nr_states) #np.array(M_inp)
        
    M_evo = np.zeros((len(traj)+1, nr_states, nr_states))
    M_evo[0, :, :] = M_TD_lambda

    # loop over trials
    for trial in range(len(traj)):
        curr_state = traj[trial][0]    
        elig_trace = np.zeros(nr_states)
        one_vector = np.zeros(nr_states)
        
        # loop over states
        for i in range(1,len(traj[trial])):
            
            old_state = curr_state
            curr_state = traj[trial][i]
            one_vector[old_state] = 1
            
            elig_trace = lambda_par*gamma*elig_trace + 1*one_vector
            pred_error = one_vector + gamma*M_TD_lambda[curr_state,:] - M_TD_lambda[old_state,:]
            
            M_TD_lambda = M_TD_lambda + alpha*np.outer(elig_trace,pred_error)
            
            one_vector[old_state] = 0 #reset to vector of all zeros
        M_evo[trial+1, :, :] = M_TD_lambda
        
    return np.array(M_evo)

def theor_TD_lambda_mod(traj, gamma, lambda_par, nr_states, alpha, bias = 1, M_TD_lambda=[]):
    
    if M_TD_lambda == []:
        M_TD_lambda = np.eye(nr_states) #np.array(M_inp)
        
    M_evo = np.zeros((len(traj), nr_states, nr_states))
    M_evo[0, :, :] = M_TD_lambda

    # loop over trials
    for trial in range(len(traj)):
        curr_state = traj[trial][0]    
        elig_trace = np.zeros(nr_states)
        one_vector = np.zeros(nr_states)
        
        # loop over states
        for i in range(1,len(traj[trial])):
            
            old_state = curr_state
            curr_state = traj[trial][i]
            one_vector[old_state] = 1*bias
            
            elig_trace = lambda_par*gamma*elig_trace + 1*one_vector
            pred_error = one_vector + gamma*M_TD_lambda[curr_state,:] - M_TD_lambda[old_state,:]
            
            M_TD_lambda = M_TD_lambda + alpha*np.outer(elig_trace,pred_error)
            
            one_vector[old_state] = 0 #reset to vector of all zeros
        M_evo[trial, :, :] = M_TD_lambda
        
    return np.array(M_evo)

    
def theor_model2(traj, gamma, nr_states, alpha, M_inp = []):
    
    if M_inp == []:
        M_inp = np.eye(nr_states) 

    M_evo = np.zeros((len(traj), nr_states, nr_states))
    M_evo[0, :, :] = M_inp

    for trial in range(len(traj)):
        M = np.zeros((nr_states, nr_states))
        Ns = np.zeros((nr_states, nr_states))    
        
        for ii in range(len(traj[trial])):
            init_state = int(traj[trial][ii])
            #M[init_state][init_state] = M[init_state][init_state] + [1]  # gamma**0
            Ns[init_state,:] = Ns[init_state,:] + 1
            #shortest_trajectory = np.zeros(nr_states) #counts number of times you visit each state jj from ii
            for jj in range(ii,len(traj[trial])):
                end_state = int(traj[trial][jj])
                M[init_state][end_state] = M[init_state][end_state] + [gamma**(jj-ii)]
                
        M = np.array(M)
        M = M_inp + alpha*(M - Ns*M_inp) #in the limit, it converges to the 
        M_inp = M 
        M_evo[trial, :, :] = M_inp

    return M_evo

def offline_MC(traj, gamma, nr_states, alpha, M_inp = []):
    
    if M_inp == []:
        M_inp = np.eye(nr_states) 

    M_evo = np.zeros((len(traj), nr_states, nr_states))
    #M_evo[0, :, :] = M_inp
    
#    M_tots =  np.zeros((len(traj)+1, nr_states, nr_states))
#    N_tots =  np.zeros((len(traj)+1, nr_states, nr_states))
#
#    M_tots[0] = M_inp
#    N_tots[0] =  np.ones((nr_states, nr_states))
#    ratios = M_inp
    
    for trial in range(len(traj)):
        M = np.zeros((nr_states, nr_states))
        Ns = np.zeros((nr_states, nr_states))    
        
        for ii in range(len(traj[trial])):
            init_state = int(traj[trial][ii])
            #M[init_state][init_state] = M[init_state][init_state] + [1]  # gamma**0
            Ns[init_state,:] = Ns[init_state,:] + 1
            #shortest_trajectory = np.zeros(nr_states) #counts number of times you visit each state jj from ii
            for jj in range(ii,len(traj[trial])):
                end_state = int(traj[trial][jj])
                M[init_state][end_state] = M[init_state][end_state] + [gamma**(jj-ii)]
                
        ratio = np.array(M)/Ns
        
        M_inp[Ns>0] = M_inp[Ns>0] + alpha*(ratio[Ns>0] - M_inp[Ns>0]) #in the limit, it converges to the 
        M_evo[trial, :, :] = M_inp

#        M_tots[trial] = M 
#        N_tots[trial] = Ns
#        
#        ratio = M_tots[:trial].sum(axis=0)/N_tots[:trial].sum(axis=0)
#        #ratio[N_tots[:trial].sum(axis=0)==0] = 0 
        
#        M_evo[trial, :, :] = ratio

    return M_evo

def mix_model12(traj, gamma, lambda_par, nr_states, alpha, ratio_replays = 0.1, M_inp=[], decay_replay = False):
    
    trials = len(traj)
    
    if M_inp == []:
        M_inp = np.eye(nr_states) 
        
    M_evo = np.zeros((trials, nr_states, nr_states))
    M_evo[0, :, :] = M_inp
    
    ratio_replays_min = ratio_replays
    if decay_replay == True: 
        ratio_replays_min = 0 
        
    ratios = np.linspace(ratio_replays, ratio_replays_min, int(trials/2))
    ratios = np.concatenate((ratios, np.zeros(int(trials/2))))
    for trial in range(trials):
        
        rnd = np.random.rand()
        #if trial%2 == 0:
        if rnd >= ratios[trial]:
            M = theor_TD_lambda([traj[trial]], gamma, lambda_par, nr_states, alpha, M_inp)
            M_inp = M[0] 
            
        #if trial%2 == 1:
        else:
            M = theor_model2([traj[trial]], gamma, nr_states, alpha, M_inp)
            M_inp = M[0] 
            
        M_evo[trial, :, :] = M_inp
        
    return M_evo 
            
def fetch_parameters(): 
    
    params = {
    
           'A_plus': 3,
           'T': 500,
           'eps0': 1,
           'eta_stdp': 0.001,
           'no_states': 4,
           'rate_ca3': 0.05,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60,
           'N_post':5,
           'N_pre_tot':10,
           'N_pre':5,
           'Trials': 300
           }
    params['eps0'] = params['eps0']/params['N_pre']
    params['pre_offset'] = params['N_pre']*.25 
    eta, lambda_var, gamma_var, bias_TD0, bias_TDlambda, bias_avg,params['A_pre'] = calculate_parameters_TD(**params)
    params['bias'] = bias_TD0
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])
    
    return params, gamma_var, eta

            
def fetch_parameters2(): 
    
    params = {
    
           'A_plus': 3,
           'T': 500,
           'eps0': 1,
           'eta_stdp': 0.001,
           'no_states': 4,
           'rate_ca3': 0.05,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60,
           'N_post':1,
           'N_pre_tot':1,
           'N_pre':1,
           'Trials': 300
           }
    params['eps0'] = params['eps0']/params['N_pre']
    params['pre_offset'] = params['N_pre']*.25 
    eta, lambda_var, gamma_var, bias_TD0, bias_TDlambda, bias_avg,params['A_pre'] = calculate_parameters_TD(**params)
    params['bias'] = bias_TD0
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])
    
    return params, gamma_var, eta, lambda_var

def fetch_parameters3(): 
    
    params = {
    
           'A_plus': 3,
           'T': 500,
           'eps0': 1,
           'eta_stdp': 0.005,
           'no_states': 1,
           'rate_ca3': 0.05,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60,
           'N_post':1,
           'N_pre_tot':1,
           'N_pre':1,
           'Trials': 400 
           #'delay': 300
           }
    params['eps0'] = params['eps0']/params['N_pre']
    params['pre_offset'] = params['N_pre']*.25 
    eta, lambda_var, gamma_var, bias_TD0, bias_TDlambda, bias_avg, params['A_pre'], params['B'] = calculate_parameters_TD(**params)
    params['bias'] = bias_TD0
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])
    params['eta'] = eta
    return params, gamma_var, eta, lambda_var

            
def fetch_parameters_new(): 
# New version with different currents 
    
    params = {
           'A_plus': 1,
           'T': 100,
           'eps0': 1,
           'eta_stdp': 0.003,
           'no_states': 4,
           'rate_ca3': 0.1,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60,
           'N_post':1,
           'N_pre_tot':1,
           'N_pre':1,
           'Trials': 400, 
           'theta': 80,
           }
#    params['eps0'] = params['eps0']/params['N_pre']
    params['pre_offset'] = 5
    eta, lambda_var, gamma_var, bias, params['A_pre'] = calculate_parameters_TD_new(**params,verbose=True)
    params['bias'] = bias
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])
    params['eta'] = eta
    return params, gamma_var, eta, lambda_var

def fetch_parameters_new_new(): 
# New version with different currents 
    
    params = {
           'A_plus': 1,
           'eps0': 1,
           'eta_stdp': 0.003,
           'no_states': 4,
           'rate_ca3': 0.1,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60,
           'N_post':1,
           'N_pre_tot':1,
           'N_pre':1,
           'Trials': 400, 
           'theta': 500,
           'delay': 20,
           'A_pre': -13.7,
           'bias': 0,
           'eta': 0

           }
#    params['eps0'] = params['eps0']/params['N_pre']
    params['T'] = params['delay'] + params['theta']
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])
    params['pre_offset'] = 1
    eta, lambda_var, gamma_var, bias = calculate_parameters_var(**params)
    params['bias'] = bias
    params['eta'] = eta
    return params, gamma_var, eta, lambda_var

def keep_gamma_eta_same(effective_connections,T, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, theta, delay, gamma_target, eta_target):
    

    A1 = rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))*N_pre 
    A2 = theta/((tau_m+tau_plus))
    LTP = eta_stdp*eps0*(A_plus*rate_ca3*tau_m*tau_plus*(A1+A2))
    # Costraint 1
    A_pre =   -(eta_target + LTP)/(eta_stdp*rate_ca3*theta)
    A_LTD = A_pre*rate_ca3*theta
    max_ltd = - A_plus*tau_m*tau_plus*(A1+A2)/theta
    A = eta_stdp*eps0*(A_plus*rate_ca3*tau_m*tau_plus*(A1+A2) + A_LTD)
    C = eta_stdp*A_plus*rate_ca3*tau_plus*(np.exp(theta/tau_plus)-1)*N_pre*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*tau_plus*(1-np.exp(-theta/tau_plus))
    # Costraint 2
    const = (1-C/A)*np.exp(-theta/tau_plus)
    delay = tau_plus*(-np.log(gamma_target) + np.log(const))
    T = theta + delay
    
    gamma_var, lambda_var, eta, bias = calculate_parameters_var(effective_connections,T, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, theta, delay)
#    eta = -A
#    lambda_var = 1/(1+C/eta)
#    gamma_var = np.exp(-T/tau_plus)/lambda_var
#    
#    D = eta_stdp*rate_ca3*A_plus*tau_plus*(1-np.exp(-theta/tau_plus))*tau_plus*(1-np.exp(-(T-theta)/tau_plus))
#    gamma_var = np.exp(-T/tau_plus)/lambda_var
#    bias = -A/D  
    b = - A_plus*tau_m*tau_plus*(rate_ca3 + 1/(tau_m+tau_plus))
    min_cond = min(max_ltd, b)
    print('Is A_pre < min? ' + str( A_pre < min_cond))
    print('=====')
    print('Lambda: '+str(lambda_var))
    print('Gamma: '+str(gamma_var))
    print('eta: '+str(eta))
    print('bias: '+str(bias))
    print('=====')
    print('A_pre: '+str(A_pre))
    print('delay: '+str(delay))
    print('T: '+str(T))
    
    return gamma_var, lambda_var, eta, A_pre, delay, T, bias
#params, gamma, eta, lambda_var = fetch_parameters_new_new()
#theor_TD_lambda(traj, gamma, lambda_var, params['no_states'], eta)

def fetch_parameters_variable_gamma(theta, delay): 
# New version with different currents 
    
    params = {
           'A_plus': 1,
           'eps0': 1,
           'eta_stdp': 0.005,
           'no_states': 4,
           'rate_ca3': 0.05,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60,
           'N_post':1,
           'N_pre_tot':1,
           'N_pre':1,
           'Trials': 400
           }
#    params['eps0'] = params['eps0']/params['N_pre']
    params['T'] = theta+ delay
    params['theta'] = theta
    params['pre_offset'] = 1
    eta, lambda_var, gamma_var, bias, params['A_pre'] = calculate_parameters_TD_new(**params)
    params['bias'] = bias
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])
    params['eta'] = eta
    return params, gamma_var, eta, lambda_var

def fetch_parameters_variable_gamma2(theta, delay): 
# New version with different currents 
    
    
    params = {
           'A_plus': 1,
           'eps0': 1,
           'eta_stdp': 0.003,
           'no_states': 4,
           'rate_ca3': 0.1,
           'step': 0.01,
           'tau_m': 2,
           'tau_plus': 60,
           'N_post':1,
           'N_pre_tot':1,
           'N_pre':1,
           'Trials': 400, 
           'theta': 80,
           'A_pre': -15
           }
#    params['eps0'] = params['eps0']/params['N_pre']
    params['T'] = theta+ delay
    params['theta'] = theta
    params['pre_offset'] = 5
    eta, lambda_var, gamma_var, bias = calculate_parameters_variable(**params)
    params['bias'] = bias
    params['effective_connections'] = create_effective_connections(params['no_states'], params['N_pre'], params['N_pre_tot'], params['N_post'])
    params['eta'] = eta
    return params, gamma_var, eta, lambda_var

#params, gamma, eta, lambda_var = fetch_parameters_variable_gamma()
def hyperbolic_discounting(effective_connections,T, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, theta, delay):
# New version with different currents 
    #theta_var = 1000
    a = A_plus*rate_ca3*tau_plus**2*tau_m
    print(a)
    b = eps0*A_pre + eps0*A_plus*tau_m*tau_plus*(1/(tau_m+tau_plus)+rate_ca3)
    c = -eps0*A_plus*tau_m*tau_plus*rate_ca3*tau_plus
    k = -np.exp(-(delay)/tau_plus)
    gamma = a*k/(b*theta+c)
    print(b)
    eta = -(b*theta+c)*rate_ca3*eta_stdp
    lambda_var = -np.exp(-theta/tau_plus)*(b*theta+c)/a
    return gamma, lambda_var, eta#(b*theta_var+c)*rate_ca3*eta_stdp

def calculate_parameters_var(effective_connections,T, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, theta, delay):
    
    T = theta + delay
    A1 = rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))*N_pre 
    A2 = theta/((tau_m+tau_plus))
    A_LTD = A_pre*rate_ca3*theta
    max_ltd = - A_plus*tau_m*tau_plus*(A1+A2)/theta
    A = eta_stdp*eps0*(A_plus*rate_ca3*tau_m*tau_plus*(A1+A2) + A_LTD)
    C = eta_stdp*A_plus*rate_ca3*tau_plus*(np.exp(theta/tau_plus)-1)*N_pre*rate_ca3*tau_m*(1-np.exp(-theta/tau_m))*tau_plus*(1-np.exp(-theta/tau_plus))
    eta = -A
    lambda_var = 1/(1+C/eta)
    gamma_var = np.exp(-T/tau_plus)/lambda_var
    
    D = eta_stdp*rate_ca3*A_plus*tau_plus*(1-np.exp(-theta/tau_plus))*tau_plus*(1-np.exp(-(T-theta)/tau_plus))
    gamma_var = np.exp(-T/tau_plus)/lambda_var
    bias = -A/D  
    b = - A_plus*tau_m*tau_plus*(rate_ca3 + 1/(tau_m+tau_plus))
    min_cond = min(max_ltd, b)
    print('Is A_pre < min? ' + str( A_pre < min_cond))
    print('=====')
    print('Lambda: '+str(lambda_var))
    print('Gamma: '+str(gamma_var))
    print('eta: '+str(eta))
    print('bias: '+str(bias))

    return gamma_var, lambda_var, eta, bias

#params, gamma, eta, lambda_var = fetch_parameters_new()

def mix_model12_seq(traj, gamma, lambda_par, nr_states, alpha, seq):
    
    trials = len(traj)
    
    M_inp = np.eye(nr_states) 
        
    M_evo = np.zeros((trials, nr_states, nr_states))
    M_evo[0, :, :] = M_inp
            
    for trial in range(trials):

        if seq[trial]==0:
            M = theor_TD_lambda([traj[trial]], gamma, lambda_par, nr_states, alpha, M_inp)
            M_inp = M[0] 
        else:
            M = theor_model2([traj[trial]], gamma, nr_states, alpha, M_inp)
            M_inp = M[0] 
            
        M_evo[trial, :, :] = M_inp
        
    return M_evo 
