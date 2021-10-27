import numpy as np

def instance_generator(seed, d=10, gamma=1, theta_ub=2, theta_lb=1e-5, t=1.05, gap_list=None):
    '''
    Generate the instance parameter list including:
    - initial points
    - reward vector theta for each coordinate
    - safety vector mu for each coordinate
    - maximum value 
    
    Input
    ------
    :param seed: seed for randomization
    :param d: dimension / total number of coordinates
    :param gamma: safety confidence threshold
    :param theta_ub: upper bound value for reward vector theta
    :param theta_lb: lower bound value for reward vector theta
    :param t: parameter to adjust the gap size. i-th coordinate has a gap value of 1-1/t^i
    :param gap_list: predefined gap value input
    
    output
    ------
    :return: - init_points
             - theta_list
             - mu_list
             - M_i_list
    
    '''
    
    np.random.seed(seed)

    mu_0 = 1
    theta_0 = 1
    
    gap_list = np.array([1 - 1/t ** i for i in range(d)]) if gap_list is None else gap_list
    init_points = np.ones(d) * 0.01
    M_i_list = np.ones(d) * 2
    
    opt_reward_0 = gamma * theta_0 / mu_0
    opt_reward_list = opt_reward_0 - gap_list

    theta_list = np.array([1] + list(np.random.uniform(theta_lb, theta_ub, d-1)))
    for i in range(d):
        if (M_i_list[i] * theta_list[i]) < opt_reward_list[i]:
            theta_list[i] = opt_reward_list[i] / M_i_list[i]
    mu_list = gamma * theta_list / opt_reward_list
    
    return init_points, theta_list, mu_list, M_i_list