import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from source.utils import TD_lambda_update, theor_trajectory

#%%

nr_states = 21
epochs = ['pre', 'post']

neg_reward = -2
replay_prob=1

tot_steps = 4000

gamma = .9
alpha = .1


#%%

def transition(M,
               elig_trace,
               reward,
               tot_steps,
               curr_state,
               prev_state,
               gamma,
               alpha,
               traj,
               replays=False,
               neg_reward = 0):

    n_states = np.shape(M)[0]

    ## REPLAYS
    if replays and curr_state == 10 and prev_state==11 and reward[0]<0:
        if np.random.rand()<replay_prob:
            M = theor_trajectory([10-x for x in range(11)],
                    n_states,
                    gamma,
                    1,
                    alpha,
                    M)
    ## SET SHOCK
    if curr_state == reward[0]:
        reward[0] = neg_reward

    ## UPDATE VALUE
    V = np.dot(M,reward)

    ## CHOOSE NEXT STATE
    possible_next = [curr_state-1] if curr_state>0 else []
    if curr_state<nr_states-1:
        possible_next.append(curr_state+1)

    probabilities = softmax([V[j] for j in possible_next])

    if np.random.rand() < probabilities[0]:
        next_state = possible_next[0]
    else:
        next_state = possible_next[1]

    traj.append(next_state)

    ## UPDATE M, e
    M, elig_trace = TD_lambda_update(
                        curr_state,
                        next_state,
                        gamma,
                        0,
                        alpha,
                        elig_trace,
                        M
                        )
    prev_state = curr_state
    curr_state = next_state

    return M, elig_trace, prev_state, curr_state, reward, traj, V

def make_traj(M,
              reward,
              tot_steps,
              curr_state,
              gamma,
              alpha,
              replays=False,
              neg_reward = 0
              ):
    traj = [curr_state]
    prev_state=curr_state

    # Initialize eligibility trace
    n_states = np.shape(M)[0]
    elig_trace = np.zeros(n_states)

    if neg_reward<0:
        # Explore until shock
        while reward[0]==0:
            M, elig_trace, prev_state, curr_state, reward, traj, V = transition(M,
                   elig_trace,
                   reward,
                   tot_steps,
                   curr_state,
                   prev_state,
                   gamma,
                   alpha,
                   traj,
                   replays=replays,
                   neg_reward=neg_reward)

        # Reset
        curr_state=20
        prev_state=20
        traj = [curr_state]

    for i in range(tot_steps):
        M, elig_trace, prev_state, curr_state, reward, traj, V = transition(M,
               elig_trace,
               reward,
               tot_steps,
               curr_state,
               prev_state,
               gamma,
               alpha,
               traj,
               replays=replays,
               neg_reward=neg_reward)

    return traj,V,M

#%%

ts = []

for tt in range(500):
    print(tt)
    all_traj = [[],[]]
    all_vals = [[],[]]
    all_M = [[],[]]

    for j,replays in enumerate([False, True]):
        reward = np.zeros(nr_states)
        curr_state = nr_states-1
        M = np.eye(nr_states)
        for t in epochs:
            n = neg_reward if t=='post' else 0
            traj,V,M = make_traj(M, reward, tot_steps, curr_state, gamma, alpha, replays=replays, neg_reward = n)

            all_traj[j].append(traj)
            all_vals[j].append(V)
            all_M[j].append(M)
    ts.append(all_traj)

#%%
# import seaborn as sns
# import pandas as pd

lw = 4
fs = 17

nstates = len(set(ts[0][0][-1]))

no_replays = []
replays = []
pr = []
for x in ts:
    p,_ = np.histogram(x[0][0], bins=nstates)
    h1,_ = np.histogram(x[0][-1], bins=nstates)
    h2,_ = np.histogram(x[1][-1], bins=nstates)
    no_replays.append(h1)
    replays.append(h2)
    pr.append(p)

no_replays = np.array(no_replays)
replays = np.array(replays)
pre = np.array(pr)

p1, ps1 = np.mean(pre, axis=0),np.std(pre, axis=0)
m1, s1 = np.mean(no_replays, axis=0),np.std(no_replays, axis=0)
m2, s2 = np.mean(replays, axis=0),np.std(replays, axis=0)

plt.figure()
plt.plot(p1, c='g', linewidth = lw, label='pre')
plt.fill_between(range(nstates), p1-ps1, p1+ps1, color='g', alpha=.2)
plt.plot(m1, c='b', linewidth = lw, label='post without replays')
plt.fill_between(range(nstates), m1-s1, m1+s1, color='b', alpha=.2)
plt.plot(m2, c='r', linewidth = lw, label='post with replays')
plt.fill_between(range(nstates), m2-s2, m2+s2, color='r', alpha=.2)
plt.legend(fontsize=fs)
plt.xticks(list(range(0,20,5)),fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel('State', fontsize=fs)
plt.ylabel('Nr of visits', fontsize=fs)
# plt.savefig('fig5_occupancy.eps',bbox_inches='tight')


#%%
fs = 60

for p in [0,1]:
    plt.figure(figsize=(10,20))
    plt.plot(all_traj[p][0],np.arange(tot_steps+1),label='pre',linewidth=lw)
    plt.plot(all_traj[p][-1],np.arange(tot_steps+1), label='post',linewidth=lw)
    r = plt.Rectangle([0,0], 10, tot_steps, fc='grey')
    r2 = plt.Rectangle([0,0], 1, tot_steps, fc='black')
    plt.gca().add_patch(r)
    plt.gca().add_patch(r2)
    plt.xlim([0,20])
    plt.ylim([tot_steps,0])
    plt.xticks(ticks=[0]+[x for x in range(4,21,5)], labels=[1]+[x+1 for x in range(4,21,5)],fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('State', fontsize=fs)
    plt.ylabel('Timestep', fontsize=fs)
    plt.legend(fontsize=fs)
    if p==0:
        plt.title('No replays',fontsize=fs)
    else:
        plt.title('Replays', fontsize=fs)
    # plt.savefig(f'fig5_replays_{p}.eps',bbox_inches='tight')

#%%
lw = 9
fs = 30

plt.figure(figsize=(10,10))
plt.plot(all_vals[0][-1], label='without replays', linewidth=lw)
plt.plot(all_vals[1][-1], label='with replays', linewidth=lw)
plt.ylabel('Value', fontsize=fs)
plt.xlabel('State', fontsize=fs)
plt.xticks(ticks=[0]+[x for x in range(4,21,5)], labels=[1]+[x+1 for x in range(4,21,5)],fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.savefig(f'fig5_value.eps',bbox_inches='tight')

#%%

fs = 40

from matplotlib import cm
top = cm.get_cmap('Greys', 128)

plt.figure(figsize=(10,10))
plt.imshow(all_M[0][-1],cmap=top,vmin=0,vmax=1)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fs)
plt.xticks([x for x in range(0,21,5)],fontsize=fs)
plt.yticks([x for x in range(0,21,5)],fontsize=fs)
plt.xlabel('Future state (CA1)',fontsize=fs)
plt.ylabel('Current state (CA3)',fontsize=fs)
plt.title('Without replays', fontsize=fs)
# plt.savefig('fig5_SR_0.eps',bbox_inches='tight')

plt.figure(figsize=(10,10))
plt.imshow(all_M[1][-1],cmap=top,vmin=0,vmax=1)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=fs)
plt.xticks([x for x in range(0,21,5)],fontsize=fs)
plt.yticks([x for x in range(0,21,5)],fontsize=fs)
plt.xlabel('Future state (CA1)',fontsize=fs)
plt.ylabel('Current state (CA3)',fontsize=fs)
plt.title('With replays', fontsize=fs)
# plt.savefig('fig5_SR_1.eps',bbox_inches='tight')

#%%
# from sklearn import manifold
# from sklearn.metrics import pairwise_distances
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# M_t = sc.fit_transform(np.transpose(M))
# Mt = np.zeros_like(M_t)
# for i in range(len(Mt)):
#     Mt[i,:] = np.concatenate([M_t[i,i:],M_t[i,:i]])


# dis_matrix = pairwise_distances(np.transpose(M), metric='euclidean')

# mds = manifold.MDS(n_components=2, random_state=0, dissimilarity='precomputed')
# mds_f = mds.fit_transform(dis_matrix)

# plt.figure()
# plt.plot(mds_f[:,0],mds_f[:,1], linewidth=lw-2, color='k', zorder=1)
# plt.scatter(mds_f[:,0],mds_f[:,1], c=list(range(21)), s=100, cmap='seismic', edgecolors='k', zorder=2)

# #%%
# ts = manifold.TSNE(n_components=2, perplexity=15, learning_rate=20)

# em = ts.fit_transform(Mt)


# plt.figure()
# plt.plot(em[:,0],em[:,1], linewidth=lw-2, color='k', zorder=1)
# plt.scatter(em[:,0],em[:,1], c=list(range(21)), s=100, cmap='seismic', edgecolors='k', zorder=2)




