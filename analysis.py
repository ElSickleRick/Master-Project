import numpy as np
import matplotlib.pyplot as plt 

class analysis:

    def __init__(self, T ,kappa, beta, pop_link_dict, eival, eivec, mu_arr):
        
        self.T = T
        self.kappa = kappa
        self.beta = beta
        self.pop_link_dict = pop_link_dict 
        self.eival = eival
        self.eivec = eivec 
        self.mu_arr = mu_arr

        return

    def Chi_abs_dist_plot(self):

        '''
        plots the distribution of the absolute values of the Chis (MF-parameters)

        Returns:
        --------
        mean: float
                mean value of distribution
        std: float
                standard deviation of distribution
        Chi_abs_arr: array of floats
                absolute values of all MF-parameters
        '''

        Chi_abs_arr = []

        for x in self.pop_link_dict:

            Chi_abs_arr.append(np.absolute(self.pop_link_dict[x][3]))

        mean = np.mean(Chi_abs_arr)
        std = np.std(Chi_abs_arr) 

        fig, ax = plt.subplots()
        ax.hist(Chi_abs_arr)

        return mean, std, Chi_abs_arr

    def free_en_calc(self):
        '''
        calculates the free energy of a given mean field solution

        Returns:
        --------
        free_en: float 
                free energy  
        '''

        F = -(1/self.beta)*np.sum(np.log(1+np.exp(-self.beta*self.eival)))
        static = 0

        for x in self.pop_link_dict:
            s, e, J, Chi = self.pop_link_dict[x]
            F += (J*(1*self.kappa/2)- (3/2)*J*self.kappa*(np.absolute(Chi))**2)*(np.absolute(Chi))**2 
            static += (J/2)*(1+self.kappa/4)

        mu_part = -np.sum(self.mu_arr)
        print("constituents of free energy density: \n static:", 2*static/((self.T+1)*(self.T+2)), "\n chemical potential:", 2*mu_part/((self.T+1)*(self.T+2)), "\n mean field:", 2*F/((self.T+1)*(self.T+2)), "\n sum:", 2*(F+static+mu_part)/((self.T+1)*(self.T+2)))


        return F + static + mu_part 






