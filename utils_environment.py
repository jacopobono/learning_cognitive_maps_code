import numpy as np 

def f_generate_traj(no_states, starts, T, max_steps):
        
    #### Sample initial position 
    in_pos = np.random.choice(no_states, 1, p=starts)[0]
    #### Generate trajectory     
    traj = [None]*max_steps
    traj[0] = in_pos

    for i in range(max_steps-1):
        
        pi=T[traj[i],:]
        traj[i+1] = np.random.choice(len(pi), 1, p=pi)[0] #use len(p) for the non absorbing state (actually no_states+1 states)
        if traj[i+1] == traj[i]:
            del traj[i+2:]
            break
    return np.array(traj)

def f_generate_n_traj(no_states, starts, T, max_steps, Trials):
    traj_tot = [] 
    for i in range(Trials):
         traj_tot.append(f_generate_traj(no_states, starts, T, max_steps))
    return traj_tot 
    

def T_open_field(no_states):

    side_len = no_states**0.5
    T = np.zeros((no_states, no_states))
        
    for i in range(no_states):
        
        neighb_states = [i-side_len,i+side_len]
        
        if (i+1)%side_len != 0:
            neighb_states = neighb_states + [i+1]
        if i%side_len != 0:
            neighb_states = neighb_states + [i-1] 
            
        neighb_states = np.array(neighb_states)
        neighb_states = neighb_states[neighb_states>=0]
        neighb_states = neighb_states[neighb_states<no_states]

        for j in neighb_states:
            T[i][int(j)] = 1 
            
    T = (np.transpose(T))/np.sum(T, axis=1)
    T = (np.transpose(T))
    
    return T  

def T_add_absorbing_states(T, eps=0.1):
    
    no_states = np.size(T,axis=0)
    
    T_new = np.zeros((no_states+1,no_states+1))
    T_new[:-1,:-1] = T
    
    to_abs_states = np.arange(no_states)
    
    for abz in to_abs_states:
        cols = T_new[abz,:]!=0
        T_new[abz, cols] -=  eps/sum(cols)
        T_new[abz,-1] = eps

    T_new[-1,-1] = 1
    
    return T_new, no_states + 1

def T_open_field_with_absorbing_state(no_states, eps=0.1):
    
    T = T_open_field(no_states)
    T = T_add_absorbing_states(T, eps)
    
    return T 

def T_random_walk_linear_track(no_states):

    T = np.array([[i == j+1 or i == j -1 for j in range(no_states)] for i in range(no_states)])*1
    # add absorbing state 
    T_new = np.zeros((no_states+1,no_states+1))
    T_new[:-1,:-1] = T
    T_new[-2:, -1] = 1
    T_new[0, -1] = 1 
    T_new = np.divide(T_new, T_new.sum(axis = 1).reshape(no_states+1, 1))    
    return T_new  

