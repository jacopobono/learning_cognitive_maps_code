import numpy as np 

def create_effective_connections(no_states, N_pre, N_pre_tot, N_post):
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

def convolution(conv, tau, X, w, step):
    conv = conv -conv/tau*step + np.multiply(X,w) - conv*(1-step/tau-np.exp(-step/tau))
    return conv

def neuron_model(epsps, eps0, step, v0, current_state):
    u = np.sum(np.sum(epsps*eps0, axis=2), axis=0) #sum over rows (all states) and pre-population 
    u[current_state, :] = u[current_state, :] + v0 #place tuned input for active trial
    X=np.random.rand(np.size(u,axis=0), np.size(u,axis=1)) < (u*step)
    return X

def stdp(A_plus, A_minus, tau_plus, tau_minus, ca1_spike_train, ca3_spike_train, conv_pre, conv_post, step):
    [no_states,no_states,N_pre]=conv_pre.shape
    N_post = np.size(conv_post,axis=2)
     
    #update traces - case with coincident spikes
    conv_pre_old = convolution(conv_pre, tau_plus, ca3_spike_train, np.zeros([no_states, no_states,N_pre]), step)
    conv_post_old = convolution(conv_post, tau_minus, ca1_spike_train, np.zeros([no_states, no_states,N_post]), step)
    
    #update traces
    conv_pre = convolution(conv_pre, tau_plus, ca3_spike_train, np.ones([no_states, no_states,N_pre]), step)
    conv_post = convolution(conv_post, tau_minus, ca1_spike_train, np.ones([no_states, no_states,N_post]), step)
    
    #expand
    conv_post_old_exp = np.tile(np.expand_dims(conv_post_old,axis=2),[1,1,N_pre,1])
    conv_post_exp = np.tile(np.expand_dims(conv_post,axis=2),[1,1,N_pre,1])
    ca1_spike_train = np.tile(np.expand_dims(ca1_spike_train,axis=2),[1,1,N_pre,1])
    
    conv_pre_old_exp = np.tile(np.expand_dims(conv_pre_old,axis=3),[1,1,1,N_post])
    conv_pre_exp = np.tile(np.expand_dims(conv_pre,axis=3),[1,1,1,N_post])
    ca3_spike_train = np.tile(np.expand_dims(ca3_spike_train,axis=3),[1,1,1,N_post])
    
    #total change in synapse due to stpd
    W = ((A_plus*conv_pre_exp*ca1_spike_train + A_minus*conv_post_exp*ca3_spike_train)*
         (ca1_spike_train+ca3_spike_train!=2)+(A_plus*conv_pre_old_exp*ca1_spike_train + 
         A_minus*conv_post_old_exp*ca3_spike_train+ 
         (A_plus+A_minus)/2)*(ca3_spike_train+ca1_spike_train==2))     
    return (W, conv_pre, conv_post)

def run_td_lambda(trajectories, effective_connections, T, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, B, offline=False):
    
    traj_len = [len(t) for t in trajectories]
    tot_cumulen = np.cumsum(traj_len)
    Trials = len(traj_len)
    T_tot = int(T*sum(traj_len)/step)
    np.random.seed() #randomize seed for parallel 
    
    #initialize weight matrix
    w_init = np.tile(np.expand_dims(np.expand_dims(np.identity(no_states), axis=2), axis=3), [1, 1, N_pre_tot, N_post])
    w = w_init 
    
#    ##################### NB: ADDED BIAS INITIAL WEIGHTS ################################
#    w = effective_connections
#    #########################################################
    
    
    #effective_connections = create_effective_connections(no_states, N_pre, N_pre_tot, N_post)
    store_w = np.zeros((sum(traj_len)+1, no_states, no_states, N_pre_tot, N_post))
    print(store_w.shape)
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
    
    pot = np.empty_like(w)*0 
    dep = np.empty_like(w)*0
    
    store_rate = np.zeros(([T_tot, no_states,N_post]))
    store_pot = np.empty_like(store_w)*0
    store_dep = np.empty_like(store_w)*0
    for i in range(T_tot):
        
        # calculate current trial and state 
        current_trial = current_trial+i//int(T*tot_cumulen[current_trial]/step) #index current trial
        idx_current_state = i//int(T/step) - int(tot_cumulen[current_trial])
        curr_state = trajectories[current_trial][idx_current_state] #index current state
        
        # CA3 
        ca3_spike_train = np.zeros([1, no_states, N_pre_tot]) #spike train from CA3 input
        ca3_spike_train[0, curr_state, :] = np.random.rand(N_pre_tot)<(rate_ca3*step) #sample Poisson spike train CA3
       
        # CA1 
        epsps = convolution(epsps, tau_m, np.tile(np.expand_dims(np.transpose(ca3_spike_train,axes=[1,0,2]),axis=3),
                                                  [1,no_states,1,N_post]), w*effective_connections, step) #epsp
        ca1_spike_train = neuron_model(epsps, eps0, step, bias, curr_state) #CA1 spike train 
        store_rate[i] = ca1_spike_train
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
        if int((i+1)%int(T/step)) == 0:
#            print('reset = ' +str(i)) 
#            print(acc_tot_dw)
            if offline == True: 
                w = w + acc_tot_dw #*np.abs(w_init-1)
                w[w<0]=0
                acc_tot_dw = 0 
                #w[curr_state, curr_state, :, :] += (eta )#- B*bias)
#                w[0,0] = 1
#                w[1,1] = 1
            store_w[(i+1)//int(T/step), :, :, :, :] = w*effective_connections
            store_pot[(i+1)//int(T/step), :, :, :, :] = pot 
            store_dep[(i+1)//int(T/step), :, :, :, :] = dep 
            pot = np.empty_like(w)*0 
            dep = np.empty_like(w)*0 
        #reset between trials 
        if ((i+1)%int(tot_cumulen[current_trial]*T/step)) == 0:
            import pdb
            
            pdb.set_trace()
            trial_nr += 1            
#            if int(100*trial_nr/Trials)%20 == 0:
#                print('\t - Trial nr '+str(trial_nr)+' of '+str(Trials))
            print(trial_nr)
#            if offline == True: 
#                w = w + acc_tot_dw#*np.abs(w_init-1)
#                w[w<0]=0
#                acc_tot_dw = 0 
            #store_w[(i+1)//int(T/step), :, :, :, :] = w*effective_connections

            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps 
            conv_pre=np.zeros([no_states,no_states,N_pre_tot])
            conv_post=np.zeros([no_states,no_states,N_post])
            
    return store_w, store_pot, store_dep, store_rate

def run_td_lambda_new(trajectories, effective_connections, T, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, theta, offline=False,ini='identity'):
    
    traj_len = [len(t) for t in trajectories]
    tot_cumulen = np.cumsum(traj_len)
    Trials = len(traj_len)
    T_tot = int(T*sum(traj_len)/step)
    np.random.seed() #randomize seed for parallel 
    
    #initialize weight matrix
    # print(ini)
    init = np.identity(no_states) if ini=='identity' else 2*np.random.rand(no_states,no_states)
    w_init = np.tile(np.expand_dims(np.expand_dims(init, axis=2), axis=3), [1, 1, N_pre_tot, N_post])
    w = w_init 
    
#    ##################### NB: ADDED BIAS INITIAL WEIGHTS ################################
#    w = effective_connections
#    #########################################################
    
    
    #effective_connections = create_effective_connections(no_states, N_pre, N_pre_tot, N_post)
    store_w = np.zeros((sum(traj_len)+1, no_states, no_states, N_pre_tot, N_post))
    print(store_w.shape)
    store_w[0]=w*effective_connections
    
    # initialize ahahah e' che io mi ansio cosi tanto a prendere l'aereo che se posso evito :D 
    ca3_spike_train = np.zeros([1, no_states,N_pre_tot]) #spike train from CA3 input 
    ca1_spike_train = np.zeros([1, no_states,N_post])
    conv_pre=np.zeros([no_states,no_states,N_pre_tot])
    conv_post=np.zeros([no_states,no_states,N_post])
    epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps     
    
    trial_nr = 0
    print('Calculating TD lambda spiking ....')

    current_trial = 0
    acc_tot_dw = 0 
    
    pot = np.empty_like(w)*0 
    dep = np.empty_like(w)*0
    
#    store_rate = np.zeros(([sum(traj_len)+1, no_states,N_post]))
#    store_pot = np.empty_like(store_w)*0
#    store_dep = np.empty_like(store_w)*0
#    
#    store_bias = np.zeros(T_tot)
    
    for i in range(T_tot):
        
        # calculate current trial and state 
        current_trial = current_trial+i//int(T*tot_cumulen[current_trial]/step) #index current trial
        idx_current_state = i//int(T/step) - int(tot_cumulen[current_trial])
        curr_state = trajectories[current_trial][idx_current_state] #index current state
        time_state = (i%int(T/step)) #time in current state 
#        if time_state == int(T/2/step) : 
#            print(i)
#            print(time_state)
        # CA3 
        ca3_spike_train = np.zeros([1, no_states, N_pre_tot]) #spike train from CA3 input
        ca3_spike_train[0, curr_state, :] = np.random.rand(N_pre_tot)<(rate_ca3*step)*(time_state<int(theta/step)) #sample Poisson spike train CA3

        # CA1 
        epsps = convolution(epsps, tau_m, np.tile(np.expand_dims(np.transpose(ca3_spike_train,axes=[1,0,2]),axis=3),
                                                  [1,no_states,1,N_post]), w*effective_connections, step) #epsp
        mod_bias = bias*(time_state>= int(theta/step))
        #store_bias[i] = (mod_bias)
        if time_state == int(theta/step): 
            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps 
            
        ca1_spike_train = neuron_model(epsps, eps0, step, mod_bias, curr_state) #CA1 spike train 
        #store_rate[(i)//int(T/step)] += ca1_spike_train
#        if ca1_spike_train.any():
#            print(i)
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
        if int((i+1)%int(T/step)) == 0:
            #epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps 
#            print('reset = ' +str(i)) 
#            print(acc_tot_dw)
            if offline == True: 
                w = w + acc_tot_dw #*np.abs(w_init-1)
                w[w<0]=0
                acc_tot_dw = 0 
                #w[curr_state, curr_state, :, :] += (eta )#- B*bias)
#                w[0,0] = 1
#                w[1,1] = 1
            store_w[(i+1)//int(T/step), :, :, :, :] = w*effective_connections
            #store_pot[(i+1)//int(T/step), :, :, :, :] = pot 
            #store_dep[(i+1)//int(T/step), :, :, :, :] = dep 

            pot = np.empty_like(w)*0 
            dep = np.empty_like(w)*0 
        #reset between trials 
        if ((i+1)%int(tot_cumulen[current_trial]*T/step)) == 0:

            trial_nr += 1            
#            if int(100*trial_nr/Trials)%20 == 0:
#                print('\t - Trial nr '+str(trial_nr)+' of '+str(Trials))
            print(trial_nr)
#            if offline == True: 
#                w = w + acc_tot_dw#*np.abs(w_init-1)
#                w[w<0]=0
#                acc_tot_dw = 0 
            #store_w[(i+1)//int(T/step), :, :, :, :] = w*effective_connections

            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps 
            conv_pre=np.zeros([no_states,no_states,N_pre_tot])
            conv_post=np.zeros([no_states,no_states,N_post])
            
    return store_w#, store_pot, store_dep, store_rate#, store_bias


def run_td_lambda_new_continuousTime(trajectories, effective_connections, T_lists, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, theta, offline=False,ini='identity'):
    
    traj_len = [len(t) for t in T_lists]
    tot_cumulen = np.cumsum(traj_len)
    Trials = len(T_lists)
    T_tot = int(sum([sum(x) for x in T_lists])/step)
    print(T_tot)
    print(step)
    np.random.seed() #randomize seed for parallel 
    
    #initialize weight matrix
    # print(ini)
    init = np.identity(no_states) if ini=='identity' else 2*np.random.rand(no_states,no_states)
    w_init = np.tile(np.expand_dims(np.expand_dims(init, axis=2), axis=3), [1, 1, N_pre_tot, N_post])
    w = w_init 
    
#    ##################### NB: ADDED BIAS INITIAL WEIGHTS ################################
#    w = effective_connections
#    #########################################################
    
    
    #effective_connections = create_effective_connections(no_states, N_pre, N_pre_tot, N_post)
    store_w = np.zeros((sum(traj_len)+1, no_states, no_states, N_pre_tot, N_post))
    print(store_w.shape)
    store_w[0]=w*effective_connections
    
    # initialize ahahah e' che io mi ansio cosi tanto a prendere l'aereo che se posso evito :D 
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
    
#    store_rate = np.zeros(([sum(traj_len)+1, no_states,N_post]))
#    store_pot = np.empty_like(store_w)*0
#    store_dep = np.empty_like(store_w)*0
#    
#    store_bias = np.zeros(T_tot)
    cumul_trials = 0
    Trial_tot_t = 0
    prev_trial = 0
    prev_state = 0
    state_counter = 0
    
    for i in range(T_tot):
        
        T = T_lists[current_trial][idx_current_state]
        Ts_trial = [t/step for t in T_lists[current_trial]]
        Trial_tot_t = sum(Ts_trial)
        
        # print(i)
        prev_trial = current_trial
        current_trial = prev_trial+int((i)//(cumul_trials+int(Trial_tot_t))) #index current trial
        
        # calculate current trial and state 
        states_trial = [y for y,x in enumerate(T_lists[current_trial]) for _ in range(int(x/step)) ]
        assert Trial_tot_t == len(states_trial), f'somthing wrong {Trial_tot_t} vs {len(states_trial)}'
        
        if prev_trial != current_trial:
            cumul_trials += Trial_tot_t
        
        # print(f'curTrial {current_trial}')
        t_trial = int(i - cumul_trials)
        
        # if prev_trial!=current_trial:
        #     print('---')
        #     print(i+1)
        #     print(cumul_trials+int(Trial_tot_t))
        #     print(cumul_trials)
        #     print(trial_nr)
        #     print(current_trial)
        #     print(f' ttrial {t_trial}, {len(states_trial)}')
        prev_state = idx_current_state
        idx_current_state = states_trial[t_trial]
        
        # print(f'idx_current_state {idx_current_state}')
        curr_state = trajectories[current_trial][idx_current_state] #index current state
        t_start_state = states_trial.index(idx_current_state)
        
        time_state = t_trial - t_start_state #time in current state 
#        if time_state == int(T/2/step) : 
#            print(i)
#            print(time_state)
        # CA3 
        ca3_spike_train = np.zeros([1, no_states, N_pre_tot]) #spike train from CA3 input
        
        if idx_current_state == 2:
            rate_ca3_s = rate_ca3#*0.5
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
            
        ca1_spike_train = neuron_model(epsps, eps0, step, mod_bias, curr_state) #CA1 spike train 
        #store_rate[(i)//int(T/step)] += ca1_spike_train
#        if ca1_spike_train.any():
#            print(i)
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
            #epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps 
#            print('reset = ' +str(i)) 
#            print(acc_tot_dw)
            if offline == True: 
                w = w + acc_tot_dw #*np.abs(w_init-1)
                w[w<0]=0
                acc_tot_dw = 0 
                #w[curr_state, curr_state, :, :] += (eta )#- B*bias)
#                w[0,0] = 1
#                w[1,1] = 1
            store_w[state_counter, :, :, :, :] = w*effective_connections
            #store_pot[(i+1)//int(T/step), :, :, :, :] = pot 
            #store_dep[(i+1)//int(T/step), :, :, :, :] = dep 
            state_counter += 1
            pot = np.empty_like(w)*0 
            dep = np.empty_like(w)*0 
        #reset between trials 
        if prev_trial != current_trial:
            

            trial_nr += 1            
#            if int(100*trial_nr/Trials)%20 == 0:
#                print('\t - Trial nr '+str(trial_nr)+' of '+str(Trials))
            print(trial_nr)
#            if offline == True: 
#                w = w + acc_tot_dw#*np.abs(w_init-1)
#                w[w<0]=0
#                acc_tot_dw = 0 
            #store_w[(i+1)//int(T/step), :, :, :, :] = w*effective_connections

            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps 
            conv_pre=np.zeros([no_states,no_states,N_pre_tot])
            conv_post=np.zeros([no_states,no_states,N_post])
     
    store_w[state_counter+1, :, :, :, :] = w*effective_connections
    
    return store_w#, store_pot, store_dep, store_rate#, store_bias


def run_MC(trajectories, temporal_between,temporal_same, effective_connections, T, step, no_states, N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, theta, eta, w = [], ini='identity',spike_noise_prob=0.2):#(trajectories,w,temporal_between,temporal_same, tau_m, Trials, pre_offset, A_pre, T, no_states, bias, 
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
                
                # print(np.shape(ca3_spike_train[0,curr_state,:]))
                # test1 = np.expand_dims(ca3_spike_train[0,curr_state,:],axis=1)
                # print(np.shape(test1))
                # test2 = np.expand_dims(test1,axis=1)
                # print(np.shape(test2))
                # tot_dw[curr_state,:,:,:] = (tot_dw[curr_state,:,:,:] + eta_stdp*A_pre*
                #                       w[curr_state,:,:,:]* np.tile(test2,[1, no_states, 1, 1]))
                
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
    
    return store_w, store_w_end

def run_td_lambda_new_short(trajectory, effective_connections, T, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, theta, w, offline=False):
    
    traj_len = len(trajectory) 
    T_tot = int(T*traj_len/step)
    np.random.seed() #randomize seed for parallel 
        
    # initialize 
    ca3_spike_train = np.zeros([1, no_states,N_pre_tot]) #spike train from CA3 input 
    ca1_spike_train = np.zeros([1, no_states,N_post])
    conv_pre=np.zeros([no_states,no_states,N_pre_tot])
    conv_post=np.zeros([no_states,no_states,N_post])
    epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps     
    
    acc_tot_dw = 0 
    
    for i in range(T_tot):
        
        # calculate current trial and state 
        idx_current_state = i//int(T/step) #- traj_len 
        curr_state = trajectory[idx_current_state] #index current state
        time_state = (i%int(T/step)) #time in current state 

        # CA3 
        ca3_spike_train = np.zeros([1, no_states, N_pre_tot]) #spike train from CA3 input
        ca3_spike_train[0, curr_state, :] = np.random.rand(N_pre_tot)<(rate_ca3*step)*(time_state<int(theta/step)) #sample Poisson spike train CA3

        # CA1 
        epsps = convolution(epsps, tau_m, np.tile(np.expand_dims(np.transpose(ca3_spike_train,axes=[1,0,2]),axis=3),
                                                  [1,no_states,1,N_post]), w*effective_connections, step) #epsp
        mod_bias = bias*(time_state>= int(theta/step))
        if time_state == int(theta/step): 
            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps 
         
            
        ca1_spike_train = neuron_model(epsps, eps0, step, mod_bias, curr_state) #CA1 spike train 

        #weight update
        [dw, conv_pre, conv_post] = stdp(A_plus, 0, tau_plus, 10, np.tile(ca1_spike_train, [no_states, 1, 1]), 
                                    np.tile(np.transpose(ca3_spike_train,axes = [1, 0, 2]), [1, no_states, 1]), conv_pre, conv_post, step)       
        tot_dw = eta_stdp*dw
        tot_dw[curr_state, :, :, :] = (tot_dw[curr_state, :, :, :] + eta_stdp*eps0*A_pre*
                                  w[curr_state, :, :, :]*np.tile(np.expand_dims(ca3_spike_train[0, curr_state, :],axis=1), [1, N_post]))
       
        if offline == True:
            acc_tot_dw += tot_dw
            #reset between states 
            if int((i+1)%int(T/step)) == 0:
                    w = w + acc_tot_dw #*np.abs(w_init-1)
                    w[w<0]=0
                    acc_tot_dw = 0 
        elif offline == False:
            w = w + tot_dw#*np.abs(w_init-1)
            w[w<0]=0

    return [np.mean(np.sum(w[:,:,:,:],axis=2)/N_pre,axis=2)] #w
           
def run_MC_raster_plot(trajectories, temporal_between,temporal_same, effective_connections, T, step, no_states, N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, theta, eta, w = []):#(trajectories,w,temporal_between,temporal_same, tau_m, Trials, pre_offset, A_pre, T, no_states, bias, 
                   # A_minus,tau_minus,A_plus,tau_plus,eta_stdp,step,N_pre_tot,N_post,N_pre,eps0,store_w,store_w_end):
                   # A_minus,tau_minus,A_plus,tau_plus,eta_stdp,step,N_pre_tot,N_post,N_pre,eps0,store_w,store_w_end):
    
    #no_states = np.size(w,axis=0)
    np.random.seed() #randomize seed for parallel 

    T = temporal_between #time in each state
    same_loc = temporal_same
    T_tot = int(T/step)
    
    if w == []:
        w_init = np.tile(np.expand_dims(np.expand_dims(np.identity(no_states), axis=2), axis=3), [1, 1, N_pre_tot, N_post])
        w = w_init*effective_connections
    
#    for traj in trajectories: 
    
    #noise_steps = 4 
    interval_range = 0.5/step

    # For loop over trials in trajectory 
    for trial in trajectories: 
        
        trial_len = len(trial)
        
        p = 0
        n_pre_spikes = np.random.binomial(1, p)+1
        if n_pre_spikes == 2: 
            n_pre_spikes = np.random.binomial(1, 0.5)*2
        n_post_spikes =np.random.binomial(1, p)+1
        if n_post_spikes == 2: 
            n_post_spikes = np.random.binomial(1, 0.5)*2
        
        times_pre = np.random.choice(np.arange(0, interval_range), n_pre_spikes)
        times_post = np.random.choice(np.arange(int(same_loc/step), interval_range + int(same_loc/step)), n_post_spikes)
        print(times_pre)
        print(times_post)
        
#        noise1 = - np.random.randint(noise_steps) 
#        noise2 = 0 
        # For loop over states in trial 
        ca3_spike_train_TOT = np.zeros([T_tot*no_states, no_states])
        ca1_spike_train_TOT = np.zeros([T_tot*no_states, no_states])


        for j in range(trial_len):
            #curr_state = trial[j] #index current state
            ca3_spike_train = np.zeros([T_tot, no_states])
            ca1_spike_train = np.zeros([T_tot, no_states])

            for i in np.arange(T_tot):
                if i in times_pre:#(i+noise1)%(T/step)==0: #or (i+1)%(T/step)==0 or (i+2)%(T/step)==0:
                    ca3_spike_train[i,j] = 1#np.random.rand() < 1
                elif i in times_post:#(((i+noise2)-int(same_loc/step))%(T/step)==0): #or (((i+1)-int(same_loc/step))%(T/step)==0) or (((i+2)-int(same_loc/step))%(T/step)==0):
                    ca1_spike_train[i,j] = 1#np.random.rand() < 1
                    
            ca3_spike_train_TOT[j*T_tot:(j+1)*T_tot] = ca3_spike_train#np.tile(np.transpose(ca3_spike_train,axes=[1,0]), [ no_states, 1])
            ca1_spike_train_TOT[j*T_tot:(j+1)*T_tot] = ca1_spike_train#np.tile(ca1_spike_train,[no_states,1])#
            
    return ca3_spike_train_TOT, ca1_spike_train_TOT

def run_TD_raster_plot(trajectories, effective_connections, T, step, no_states, 
                  N_pre, N_pre_tot, N_post, rate_ca3, eps0, bias, A_plus, 
                   tau_plus, eta_stdp, A_pre, tau_m, Trials,
                  pre_offset, eta, theta, offline=False):
    trajectory = trajectories[0]
    traj_len = len(trajectory) 
    T_tot = int(T*traj_len/step)
    np.random.seed() #randomize seed for parallel 
        
    # initialize 
    ca3_spike_train_tot = np.zeros([T_tot, no_states]) #spike train from CA3 input 
    ca1_spike_train_tot = np.zeros([T_tot, no_states])
    epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps     
    w_init = np.tile(np.expand_dims(np.expand_dims(np.identity(no_states), axis=2), axis=3), [1, 1, N_pre_tot, N_post])
    w = w_init 

    for i in range(T_tot):
        
        # calculate current trial and state 
        idx_current_state = i//int(T/step) - traj_len 
        curr_state = trajectory[idx_current_state] #index current state
        time_state = (i%int(T/step)) #time in current state 

        # CA3 
        ca3_spike_train = np.zeros([1, no_states, N_pre_tot]) #spike train from CA3 input
        ca3_spike_train[0, curr_state, :] = np.random.rand(N_pre_tot)<(rate_ca3*step)*(time_state<int(theta/step)) #sample Poisson spike train CA3
        ca3_spike_train_tot[i] = ca3_spike_train[:, :, 0].squeeze()
        
        # CA1 
        epsps = convolution(epsps, tau_m, np.tile(np.expand_dims(np.transpose(ca3_spike_train,axes=[1,0,2]),axis=3),
                                                  [1,no_states,1,N_post]), w*effective_connections, step) #epsp
        mod_bias = bias*(time_state>= int(theta/step))
        if time_state == int(theta/step): 
            epsps = np.zeros((no_states, no_states,N_pre_tot,N_post)) #epsps 
         
            
        ca1_spike_train = neuron_model(epsps, eps0, step, mod_bias, curr_state) #CA1 spike train 
        ca1_spike_train_tot[i] = ca1_spike_train[:, :].squeeze()

    return ca3_spike_train_tot, ca1_spike_train_tot
