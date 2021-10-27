import numpy as np

class Algo(object):
    '''
    The Safe-Linear Algorithm.
    '''
    
    def __init__(self, bandit, opt_reward_list):
        '''
        To initialize, create a bandit instance, compute the opt_reward_list.
        -------
        Sample:
        bandit = MAB(dim, seed=seed, init_points=init_points, theta_list=theta_list, mu_list=mu_list, M_i_list=M_i_list, s_mu=0.5, s_theta=0.5)
        opt_reward_list = [min(bandit.gamma / bandit.mu_list[i], bandit.M_i_list[i]) * bandit.theta_list[i] for i in range(bandit.d)]
        algo = Algo(bandit, opt_reward_list)
        
        -------
        To invoke the algorithm to run, use the .run_algo() function.
        '''
        super().__init__()
        
        self.iter = 1
        self.bandit = bandit
        
        ## Compute the optimal arm, its optimal value and gap values
        self.opt = [np.where(opt_reward_list == max(opt_reward_list)), max(opt_reward_list)]
        self.gap_list = [max(opt_reward_list) - min(bandit.gamma / bandit.mu_list[i], bandit.M_i_list[i]) * bandit.theta_list[i] for i in range(bandit.d)]
        self.a_safe_list = bandit.init_points # Current a_safe (at iter -1)
        self.new_unc_list = np.ones(bandit.d) # new a_unsafe (at iter)
        self.new_safe_hat_list = np.ones(bandit.d) # new a_safe (at iter)
        self.mu_hat_list = np.ones(bandit.d) # mu estiamte
        self.theta_hat_list =np.ones(bandit.d) # theta estimate
        self.N_list = np.ones(bandit.d) # number of pulls per arm
        self.termination_list = np.zeros(bandit.d) # stores the suboptimal arms with 1 at index i indicating suboptimality
        self.total_pulls = np.zeros(bandit.d) # cumulated pulls per arm
        self.opt_reward_list = opt_reward_list
        self.unsafe_count = 0 # stat: number of unsafe arms pulled
        
        ## Regret utility data
        self.simple_regret = [] # stat: regret
        self.pulls_temp = np.zeros(bandit.d) # stat: current amount of pulls

    def compute_next_safe(self, i):
        '''
        Compute the next largest safe value of arm i.
        '''
        
        ## Stop updating the verified suboptimal arm
        if self.termination_list[i]:
            return self.a_safe_list[i]
        else: 
            a_safe_new = max(self.bandit.init_points[i], self.bandit.gamma / (self.mu_hat_list[i] + 2**(1-self.iter) / self.a_safe_list[i]))
            return min(self.bandit.M_i_list[i], a_safe_new) # Restrict the value to boundary

    def compute_next_min_uncertain(self, i):
        '''
        Compute the next smallest unsafe value of arm i.
        '''
        
        ## Stop updating the verified suboptimal arm
        if self.termination_list[i]:
            return self.a_safe_list[i]
        else: 
            a_unc_new = self.bandit.gamma / max(self.mu_hat_list[i] - 2**(1-self.iter) / self.a_safe_list[i], 1e-3)
            return min(self.bandit.M_i_list[i], a_unc_new) # Restrict the value to boundary

    def compute_next_N(self):
        '''
        Compute the amount of pulls for a candidate arm in the new iteration.
        '''
        return np.ones(self.bandit.d) * np.ceil(self.bandit.s_theta**2 * 2 * 2**(2*self.iter-2) * np.log(self.bandit.d * self.iter**2 / self.bandit.delta))

    def run_one_eval(self, i): 
        '''
        Perform one round of arm pulls
        '''
        cost = self.a_safe_list[i] * self.bandit.mu_list[i] + np.random.normal(0, self.bandit.s_mu)
        reward = self.a_safe_list[i] * self.bandit.theta_list[i] + np.random.normal(0, self.bandit.s_theta)

        if self.a_safe_list[i] * self.bandit.mu_list[i] > self.bandit.gamma:
            self.unsafe_count += 1
        
        return cost, reward

    def compute_estimates(self, i):
        '''
        Compute the estimated safety and reward parameters mu_hat/theta_hat of arm i
        using current round's evaluation.
        '''
        cum_cost = 0
        cum_reward = 0
        n = self.N_list[i]
        if self.termination_list[i]:
            return self.mu_hat_list[i], self.theta_hat_list[i]
        self.total_pulls[i] += n
        for j in range(int(n)):
            cost, reward = self.run_one_eval(i)
            cum_cost += cost
            cum_reward += reward
                
        mu_hat = cum_cost / (n * self.a_safe_list[i])
        theta_hat = cum_reward / (n * self.a_safe_list[i])
        return mu_hat, theta_hat

    def check_termination(self):
        '''
        Verify if any arm is suboptimal (and thus stop pulling that arm) if:
        The arm's unsafe reward value is below the largest safe reward value
        '''
        max_safe_reward = max(self.new_safe_hat_list*(self.theta_hat_list - 2**(1-self.iter)/self.a_safe_list) * (1- self.termination_list))
        unc_reward_list = self.new_unc_list*(self.theta_hat_list + 2**(1-self.iter)/self.a_safe_list) * (1- self.termination_list)
        return unc_reward_list < max_safe_reward

    def run_algo(self):
        '''
        Run the algorithm, compute the optimizer returned and the simple regret along the process.
        '''
        np.random.seed(self.bandit.seed)
        done = sum(self.termination_list) >= self.bandit.d-1

        while not done: # To produce a valid regret value set, set it to `while self.iter < 8000:`
            self.N_list = self.compute_next_N()

            for j in range(int(max(self.N_list))):
                for i in range(len(self.a_safe_list)):
                    if self.termination_list[i]:
                        continue
                    cost, reward = self.run_one_eval(i)
                    self.pulls_temp[i] += 1
                self.simple_regret += [self.compute_regret()]
                    
            for i in range(len(self.a_safe_list)):
                mu_hat, theta_hat =  self.compute_estimates(i)
                self.mu_hat_list[i] = mu_hat
                self.theta_hat_list[i] = theta_hat
                
                self.new_safe_hat_list[i] = self.compute_next_safe(i)
                self.new_unc_list[i] = self.compute_next_min_uncertain(i)

            self.termination_list = self.check_termination()
            self.a_safe_list = self.new_safe_hat_list
            self.iter += 1

            done = sum(self.termination_list) >= self.bandit.d-1

        # return proper cases
        self.curr_best = [np.where(self.termination_list == 0), self.iter]
    
        print(f'seed {self.bandit.seed} dim {self.bandit.d} completed')
        return np.where(self.termination_list == 0), self.iter

    def check_safety(self):
        '''
        Check if the entire set of arms pulled are safe. (This differs from the check_safety in MAB class
        which examines an individual arm's value).
        '''
        return self.bandit.gamma - self.a_safe_list * self.bandit.mu_list

    def compute_regret(self):
        '''
        Compute simple regret at current arm pull.
        '''
        max_val = -1
        max_coor = -1
        max_a_val = -1
        for i in range(self.bandit.d):
            curr_theta = self.theta_hat_list[i]
            a_safe_new = self.a_safe_list[i]
            if curr_theta * a_safe_new > max_val:
                max_coor = i
                max_val = a_safe_new * curr_theta
                max_a_val = a_safe_new

        return min(max_a_val * self.bandit.theta_list[max_coor], 1), sum(self.pulls_temp)
        
    def return_key_stats(self):
        '''
        Returning key statistics
        '''
        ## dimension
        res = {}
        res['dim'] = self.bandit.d
        
        ## gap
        res['gap'] = self.gap_list
        res['min_gap'] = min(self.gap_list[1:]) # change this implementation if your optimal arm is not 0
        
        ## total samples pulled
        res['total arm pulls'] = sum(self.total_pulls)
        
        ## unsafe pulls
        res['unsafe pulls'] = self.unsafe_count
        
        ## optimal group
        res['true optimal'] = self.opt[0][0][0]
        res['current optimal'] = self.curr_best[0][0][0]
        
        return res