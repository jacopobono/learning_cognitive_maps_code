import numpy as np
import matplotlib.pyplot as plt
# import os


fs = 16

Ts = [50, 100, 200, 400]
biases = {}
gammas = {}
lambdas = {}

for T in Ts:

    alphas = np.arange(0,1,0.1)
    alphas[0] += 0.01

    A_plus= 1
    eta_stdp= 0.003 #0.002
    rate_ca3= 0.1
    step= 0.01
    tau_m= 2
    tau_plus= 60 #20
    N_post=1
    N_pre_tot=1
    N_pre=250
    eps0 = 1/N_pre

    # Total time in a state
    # T = 100

    theta = alphas*T
    delay = (1-alphas)*T

    # Calculation for parameter A
    A1 = N_pre*rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))
    # A1 = 1*rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))
    A2 = theta/((tau_m+tau_plus))

    # Depression amplitude
    max_ltd = - A_plus*tau_m*tau_plus*(A1+A2)/theta
    A_pre = -11
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

    a = (A_plus*eps0*tau_m*tau_plus*(rate_ca3+1/(tau_plus+tau_m)) + A_pre)
    b = -1*A_plus*eps0*rate_ca3*tau_m*tau_plus**2

    gamma_approx = np.exp(-delay/tau_plus)/(1+a*theta/b)

    # Bias current
    bias = -A/D

    biases[T] = bias
    gammas[T] = gamma_var
    lambdas[T] = lambda_var

figa = plt.figure()
plt.plot(alphas, np.transpose(list(biases.values())), label=biases.keys())
plt.title('CA1 place-tuned input', fontsize=fs)
plt.ylabel(r'$\rho^{bias}$', fontsize=fs)
plt.xlabel(r'$\theta$ / T', fontsize=fs)
leg = plt.legend(title='T', fontsize=fs)
leg.get_title().set_fontsize(fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

figb = plt.figure()
plt.plot(alphas, np.transpose(list(gammas.values())), label=gammas.keys())
plt.title('Gamma', fontsize=fs)
plt.ylabel(r'$\gamma$', fontsize=fs)
plt.xlabel(r'$\theta$ / T', fontsize=fs)
leg = plt.legend(title='T', fontsize=fs)
leg.get_title().set_fontsize(fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

figc = plt.figure()
plt.plot(alphas, np.transpose(list(lambdas.values())), label=lambdas.keys())
plt.title('Lambda', fontsize=fs)
plt.ylabel(r'$\lambda$', fontsize=fs)
plt.xlabel(r'$\theta$ / T', fontsize=fs)
leg = plt.legend(title='T', fontsize=fs)
leg.get_title().set_fontsize(fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)



#%%

A_pres = [-7, -9, -11]
biases = {}
gammas = {}
lambdas = {}

for A_pre in A_pres:

    alphas = np.arange(0,1,0.1)
    alphas[0] += 0.01

    A_plus= 1
    eta_stdp= 0.003 #0.002
    rate_ca3= 0.1
    step= 0.01
    tau_m= 2
    tau_plus= 60 #20
    N_post=1
    N_pre_tot=1
    N_pre=250
    eps0 = 1/N_pre

    # Total time in a state
    T = 100

    theta = alphas*T
    delay = (1-alphas)*T

    # Calculation for parameter A
    A1 = N_pre*rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))
    # A1 = 1*rate_ca3*(1-np.exp(-theta/tau_m))*(theta-tau_plus*(1-np.exp(-theta/tau_plus)))
    A2 = theta/((tau_m+tau_plus))

    # Depression amplitude
    max_ltd = - A_plus*tau_m*tau_plus*(A1+A2)/theta
    # A_pre = -11 #-13 #max_ltd + 35/max_ltd # max_ltd - 15
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

    a = (A_plus*eps0*tau_m*tau_plus*(rate_ca3+1/(tau_plus+tau_m)) + A_pre)
    b = -1*A_plus*eps0*rate_ca3*tau_m*tau_plus**2

    gamma_approx = np.exp(-delay/tau_plus)/(1+a*theta/b)

    # Bias current
    bias = -A/D

    biases[A_pre] = bias
    gammas[A_pre] = gamma_var
    lambdas[A_pre] = lambda_var



figd = plt.figure()
plt.plot(alphas, np.transpose(list(biases.values())), label=biases.keys())
plt.title('CA1 place-tuned input', fontsize=fs)
plt.ylabel(r'$\rho^{bias}$', fontsize=fs)
plt.xlabel(r'$\theta$ / T', fontsize=fs)
leg = plt.legend(title=r'$A_{pre}$', fontsize=fs)
leg.get_title().set_fontsize(fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

fige = plt.figure()
plt.plot(alphas, np.transpose(list(gammas.values())), label=gammas.keys())
plt.title('Gamma', fontsize=fs)
plt.ylabel(r'$\gamma$', fontsize=fs)
plt.xlabel(r'$\theta$ / T', fontsize=fs)
plt.ylim([0,1])
leg = plt.legend(title=r'$A_{pre}$', fontsize=fs)
leg.get_title().set_fontsize(fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

figf = plt.figure()
plt.plot(alphas, np.transpose(list(lambdas.values())), label=lambdas.keys())
plt.title('Lambda', fontsize=fs)
plt.ylabel(r'$\lambda$', fontsize=fs)
plt.xlabel(r'$\theta$ / T', fontsize=fs)
leg = plt.legend(title=r'$A_{pre}$', fontsize=fs)
leg.get_title().set_fontsize(fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)



