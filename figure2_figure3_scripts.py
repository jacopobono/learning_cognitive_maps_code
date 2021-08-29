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

if __name__ == '__main__':

        
    # comment_script = input("Comment: ")

    plt.close('all')
    
    newpath = str(datetime.datetime.now()) 
    newpath= newpath.translate(str.maketrans(string.punctuation, '_'*len(string.punctuation)))
    newpath=newpath.replace(" ", "_")
    
    #store comment about experiment 
    # g = open(newpath+'/comment.txt','w')
    # g.write(comment_script)
    # g.close()
    
    # #store name experiment in file
    # f = open("list_experiments", "a")
    # f.write(newpath+ "\n")
    # f.close()
    
    #store copy of the script 
    # filename = __file__
    # filename = filename.split("/")1
    # filename = filename[-1]
    # shutil.copy(filename, newpath)
    
    # traj = pickle.load(open('1000trajectories.pkl', 'rb'))
    
    ini_run = 0
    fin_run = 10
    assert fin_run>ini_run, 'should have more than 1 run!'
    tot_runs = fin_run-ini_run
    tot_epochs = 50
    
    
    # traj = [[t[:] for t in x[:tot_epochs]] for x in traj[ini_run:fin_run]]
    #traj = [t[:] for t in traj[:1][:tot_epochs]]*tot_runs 
    
    # traj_linear_track = np.repeat([np.repeat(np.arange(0, 1).reshape(1,-1), tot_epochs, axis=0)], tot_runs, axis=0)
    traj_linear_track = np.repeat([np.repeat([[0, 1, 2, 3]], tot_epochs, axis=0)], tot_runs, axis=0)
    traj = traj_linear_track
    T_list = [[100,100,100,100]]*tot_epochs
    
    
    mul_proc = True
    mode = 'MC'
    
    if mode == 'TD':
        fun_to_run = parall_fun
        newpath = 'TD_runs/'+newpath
    elif mode == 'MC':
        fun_to_run = parall_fun_MC
        newpath = 'MC_runs/'+newpath
        
    ### make directory
    os.makedirs(newpath)
    
    if mul_proc:
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
    
    
