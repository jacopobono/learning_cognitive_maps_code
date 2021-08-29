import numpy as np
import pylab as plt 
from utils_learning_rules import fetch_parameters_new
from utils_environment import T_random_walk_linear_track
from utils_nn import run_td_lambda_new
import pickle 
import multiprocessing
from multiprocessing import Pool 
from main_compareMLmodel12_random_walk_parall import root_mse
import datetime, os, string

def parall_fun(traj):
    
    traj,nr,ini,path = traj
    print(nr)
    
    params, gamma, eta, lambda_var = fetch_parameters_new()
    params['trajectories'] = traj 
    params['offline'] = False
    
    # print(gamma)
    M_stdp = run_td_lambda_new(**params)#,ini='random')    
    # M_TD = theor_TD_lambda(traj, gamma, lambda_var, params['no_states'], eta)

    with open(path+f'/TD_traj_{ini+nr}.pkl','wb') as f:
        pickle.dump(M_stdp,f)
        
    return M_stdp, np.zeros_like(M_stdp) #M_TD

if __name__ == '__main__':

        
    # comment_script = input("Comment: ")

    plt.close('all')
    
    newpath = str(datetime.datetime.now()) 
    newpath= newpath.translate(str.maketrans(string.punctuation, '_'*len(string.punctuation)))
    newpath=newpath.replace(" ", "_")
    newpath = 'TD_runs/'+newpath
    
    ### make directory
    os.makedirs(newpath)
    
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
    
    traj = pickle.load(open('1000trajectories.pkl', 'rb'))
    
    ini_run = 0
    fin_run = 2
    assert fin_run>ini_run, 'should have more than 1 run!'
    tot_runs = fin_run-ini_run
    tot_epochs = 75
    
    
    traj = [[t[:] for t in x[:tot_epochs]] for x in traj[ini_run:fin_run]]
    #traj = [t[:] for t in traj[:1][:tot_epochs]]*tot_runs 
    
#    traj_linear_track = np.repeat([np.repeat(np.arange(0, 1).reshape(1,-1), tot_epochs, axis=0)], tot_runs, axis=0)
    # traj_linear_track = np.repeat([np.repeat([[0, 1, 2, 3]], tot_epochs, axis=0)], tot_runs, axis=0)
    # traj = traj_linear_track
    
    cc = int(.9*multiprocessing.cpu_count())
    pool = Pool(cc) 
    
    output = pool.map(parall_fun, zip(traj,range(len(traj)),[ini_run]*len(traj),[newpath]*len(traj) ) )
    # print(np.shape(output))
    output=np.array(output)
    outputs = output[:,0]
    
    # pickle.dump(output, open(newpath + '/output_stdp.pkl','wb'))
    #pickle.dump(output_td, open(newpath + '/output_td.pkl','wb'))
    
    l = [np.cumsum([len(x) for x in t]) for t in traj]
    w_stdps = np.array([[outputs[i][l_i] for l_i in l[i]] for i in range(len(outputs))])
    # pickle.dump(w_stdps, open(newpath + '/w_stdps.pkl','wb'))
    # asssdassd
    params, gamma_var, _, _ = fetch_parameters_new()
    w_stdps2 = np.mean(np.sum(w_stdps[:,:,:,:,:,:],axis=-2)/params['N_pre'],axis=-1)
    # pickle.dump(w_stdps2, open(newpath + '/w_stdps2.pkl','wb'))
    
    init = np.tile(np.expand_dims(np.expand_dims(np.eye(4),0),0),[tot_runs,1,1,1])
    w_stdps2 = np.concatenate([init,w_stdps2],axis=1)
    
    T_mat = T_random_walk_linear_track(params['no_states']-1)
    M_theor = np.linalg.inv(np.eye(params['no_states']-1)-gamma_var*T_mat[:-1, :-1])
    #
    fig = plt.figure()
    for w1 in range(params['no_states']-1):
        for w2 in range(params['no_states']-1):
            if 3 not in [w1,w2]:
                plt.plot(w_stdps2[:, :, w1, w2].mean(axis=0))
                plt.plot([0,len(w_stdps2[0,:,0,0])],[M_theor[w1, w2],M_theor[w1, w2]])
    # fig.savefig(newpath + '/weights.png')
    
    fig2 = plt.figure()
    root_mse_TD_spiking,std = root_mse(w_stdps2[:,:,:-1,:-1], M_theor, 0)
    plt.plot(root_mse_TD_spiking)    
    plt.fill_between(range(len(std)), root_mse_TD_spiking-std, root_mse_TD_spiking+std,alpha=.3)
    # fig2.savefig(newpath + '/MSE_TD_spiking.png')
    # pickle.dump(root_mse_TD_spiking, open('rmse_TD_id.pkl','wb'))
    
    fig3 = plt.figure()
    w_stdp_matrix = plt.imshow(w_stdps2[:,-1,:,:].mean(axis=0)) 
    # fig3.savefig(newpath + '/weights_matrix.png')
