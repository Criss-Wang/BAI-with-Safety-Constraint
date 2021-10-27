import numpy as np

class MAB(object):
    ## TODO: Set s_mu and s_theta to depend on the coordinates
    '''
    A multi-arm bandit class with possible discretization and safety guarantee checks.
    '''
    def __init__(self, d=10, delta=0.01, s_mu=0.1, s_theta=0.1, gamma=1, seed=42, 
                 init_points=None,
                 mu_list=None, 
                 theta_list=None,
                 M_i_list=None
                ):
        '''
        Initialize a multi-arm bandit instance.
        
        :param d: size of bandit instance (default=10)
        :param delta: confidence threshold (default=0.01)
        :param s_mu: standard deviation of noise for cost evaluation (default=0.1)
        :param s_theta: standard deviation of noise for reward evaluation (default=0.1)
        :param gamma: threshold (default=1)
        :param seed: seed to fix sequence output (default=42)
        :param init_points: The set of initial points (default=np.ones(d) * 0.1)
        :param mu_list: The set of safety parameters (default=np.array([1] + [1.5] * (d-1)))
        :param theta_list: The set of reward parameters (default=np.array([1] + [0.5] * (d-1)))
        :param M_i_list: The set of boundary values (default=np.ones(d) * 2)
        '''
        super().__init__()
        self.d = d
        self.delta = delta
        self.s_mu = s_mu
        self.s_theta = s_theta
        self.gamma = gamma
        self.seed = seed

        self.init_points = init_points if init_points is not None else np.ones(d) * 0.1
        self.mu_list = mu_list if mu_list is not None else np.array([1] + [1.5] * (d-1))
        self.theta_list = theta_list if theta_list is not None else np.array([1] + [0.5] * (d-1))
        self.M_i_list = M_i_list if M_i_list is not None else np.ones(d) * 2

    def get_bounds(self):
        '''
        Return the value bounds [a_{0,i}, M_i] of each arm in the instance.
        '''
        return [(lb, ub) for lb, ub in np.vstack((self.init_points, self.M_i_list)).T]

    def compute_discretization(self, n_samples=100):
        '''
        Compute a discretized arm value sets for any algorithms that accept only finite value choices.
        Each arm's value is discretized in n_samples evenly distributed in [a_{0,i}, M_i].
        For instance, if n_samples = 3, then each arm in [1,3] has 3 values: 1,2,3.
        
        :param n_samples: number of sample values of an arm
    
        -------------
        :return: d x n_samples x d tensor with n_samples of R^d vectors represents discretized values of each arm.
        '''
        
        bounds = self.get_bounds()
        param_set = None
        d = len(bounds)
        for i in range(d):
            bound = bounds[i]
            a = np.zeros(d)
            a[i] = 1.0
            a = [a] * n_samples
            i_val = np.diag(np.linspace(bound[0],bound[1],n_samples)).dot(np.array(a))
            if param_set is None:
                param_set = i_val
            else:
                param_set = np.vstack((param_set, i_val))
        return np.atleast_2d(param_set)
    
    def check_safety(self, x):
        '''
        Check if an arm's value is safe.
        '''
        
        pos = x>0
        return (x * self.mu_list)[pos] < self.gamma