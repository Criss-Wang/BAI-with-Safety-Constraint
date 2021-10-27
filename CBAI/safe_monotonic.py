import numpy as np
import numpy.linalg as la
import numpy.random as rand
import copy
import time


'''
Credit: Andrew Wagenmaker - University of Washington
Implementation of Safe-Monotonic for experimental trials
'''

def sigmoid(x,a,b):
    '''
    Simple Sigmoid function to simulate clinical drug dosage response model.

    :param x: drug dosage
    :param a: scale of dosage
    :param b: offset
    '''
    return 1/(1+np.exp(-a*(x)-b))

def safe_monotonic_bai(a0,gamma,delta,epssafe,f_func,g_func,f_params_b1,f_params_b0, g_params_b1,g_params_b0,noise_var=1):
    '''
    :param thetast: R^d reward vector
    :param must: R^d safety vector
    :param a0: R^d vector containing initial safe values
    :param gamma: safety constraint value
    :param delta: confidence
    :param epssafe: tolerance above gamma we allow sampling at
    :param f_func: length d list of reward functions
    :param g_func: length d list of cost functions
    :param f_params_b1: length d list of reward function parameters (corresponding to `a` in sigmoid)
    :param f_params_b0: length d list of reward function parameters (corresponding to `b` in sigmoid)
    :param g_params_b1: length d list of cost function parameters (corresponding to `a` in sigmoid)
    :param g_params_b0: length d list of cost function parameters (corresponding to `b` in sigmoid)
    :param noise_var: noise variance (default is 1)

    :return: T - number of samples taken
             best_arm - estimate of best arm
             unsafe - integer count of number of unsafe arms pulled
             regret - regret statistics
    '''

    d = len(a0)
    asafe_max = copy.copy(a0)
    aunsafe_max = copy.copy(a0)
    T = 0
    epoch_idx = 1
    unsafe = 0
    active = np.ones(d)
    found_unsafe = np.zeros(d)
    reward = []

    fhat = np.zeros(d)
    fhat_u = 1000*np.ones(d)
    ghat = np.zeros(d)
    ghat_u = np.zeros(d)
    t = 1

    while np.sum(active) > 1 or T < 10000:
        epsilon_l = 2**(-epoch_idx)
        val = -1
        for i in range(d):
            '''
            Run evaluations only for candidate arms.
            Note that here we simulate n trials by directly applying `np.sqrt(noise_var/n)*rand.randn()`.
            This is because the empirical mean is simply the true mean value + empirical noise average
            where the noise ~ N(0, noise_var).
            '''
            if active[i]:
                n = np.ceil(noise_var*2*np.log(8*(t**2)/delta)*2**(2*epoch_idx))
                fhat[i] = f_func[i](asafe_max[i],f_params_b1[i],f_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                ghat[i] = g_func[i](asafe_max[i],g_params_b1[i],g_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                T += n
                t += 1
                
                if g_func[i](asafe_max[i],g_params_b1[i],g_params_b0[i]) > gamma + epssafe:
                    unsafe += n
                
                while gamma - ghat[i] > 2*epsilon_l:
                    asafe_max[i] = gamma + asafe_max[i] - ghat[i] - epsilon_l
                    n = np.ceil(noise_var*2*np.log(8*(t**2)/delta)*2**(2*epoch_idx))
                    fhat[i] = f_func[i](asafe_max[i],f_params_b1[i],f_params_b0[i]) +  np.sqrt(noise_var/n)*rand.randn()
                    ghat[i] = g_func[i](asafe_max[i],g_params_b1[i],g_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                    T += n
                    t += 1
                    
                    if g_func[i](asafe_max[i],g_params_b1[i],g_params_b0[i]) > gamma + epssafe:
                        unsafe += n
                if found_unsafe[i] == 0:
                    n = np.ceil(noise_var*2*np.log(8*(t**2)/delta)*2**(2*epoch_idx))
                    fhat_u[i] = f_func[i](aunsafe_max[i],f_params_b1[i],f_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                    ghat_u[i] = g_func[i](aunsafe_max[i],g_params_b1[i],g_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                    T += n
                    t += 1

                    if g_func[i](aunsafe_max[i],g_params_b1[i],g_params_b0[i]) > gamma + epssafe:
                        unsafe += n

                    while gamma + epssafe - ghat_u[i] > 2*epsilon_l:
                        aunsafe_max[i] = gamma + epssafe + aunsafe_max[i] - ghat_u[i] - epsilon_l
                        n = np.ceil(noise_var*2*np.log(8*(t**2)/delta)*2**(2*epoch_idx))
                        fhat_u[i] = f_func[i](aunsafe_max[i],f_params_b1[i],f_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                        ghat_u[i] = g_func[i](aunsafe_max[i],g_params_b1[i],g_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                        T += n
                        t += 1

                        if g_func[i](aunsafe_max[i],g_params_b1[i],g_params_b0[i]) > gamma + epssafe:
                            unsafe += n

                        if ghat_u[i] - epsilon_l >= gamma:
                            found_unsafe[i] = 1
                            break
                else:
                    aunsafe_max0 = aunsafe_max[i]
                    ghati_u0 = ghat_u[i]
                    fhati_u0 = fhat_u[i]

                    aunsafe_max[i] = aunsafe_max0/2 + asafe_max[i]/2
                    n = np.ceil(noise_var*2*np.log(8*(t**2)/delta)*2**(2*epoch_idx))
                    fhat_u[i] = f_func[i](aunsafe_max[i],f_params_b1[i],f_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                    ghat_u[i] = g_func[i](aunsafe_max[i],g_params_b1[i],g_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                    T += n
                    t += 1

                    while ghat_u[i] - epsilon_l < gamma:
                        aunsafe_max[i] = aunsafe_max0/2 + aunsafe_max[i]/2
                        if aunsafe_max0 - aunsafe_max[i] <= epsilon_l:
                            aunsafe_max[i] = aunsafe_max0
                            ghat_u[i] = ghati_u0
                            fhat_u[i] = fhati_u0
                            break

                        n = np.ceil(noise_var*2*np.log(8*(t**2)/delta)*2**(2*epoch_idx))
                        fhat_u[i] = f_func[i](aunsafe_max[i],f_params_b1[i],f_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                        ghat_u[i] = g_func[i](aunsafe_max[i],g_params_b1[i],g_params_b0[i]) + np.sqrt(noise_var/n)*rand.randn()
                        T += n
                        t += 1
            if sigmoid(asafe_max[i],f_params_b1[i],f_params_b0[i]) > val:
                val = sigmoid(asafe_max[i],f_params_b1[i],f_params_b0[i])
            for i in range(d):
                if found_unsafe[i] == 1 and fhat_u[i] < np.max(fhat) - 2*epsilon_l:
                    active[i] = 0
        epoch_idx += 1
        ## Compute the current best reward value for regret statistics latter
        reward += [[val, T]]

    return T, np.argmax(active), unsafe, epoch_idx, reward

