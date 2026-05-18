import numpy as np
import matplotlib.pyplot as plt 

class analysis:

    def __init__(self, T ,kappa, beta,  pop_link_dict, eival, eivec, mu_hist_dict, mu_arr, bond_hist_dict, sc_iter):
        
        self.T = T
        self.kappa = kappa
        self.beta = beta
        self.pop_link_dict = pop_link_dict 
        self.eival = eival
        self.eivec = eivec
        self.mu_hist_dict = mu_hist_dict
        self.mu_arr = mu_arr
        self.bond_hist_dict = bond_hist_dict
        self.sc_iter = sc_iter

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
        calculates the free energy desnsity of a given mean field solution

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


        return 2*(F + static + mu_part)/((self.T+1)*(self.T+2))



    def MF_iter_plot(self):

        fig, ax = plt.subplots(2,2)
        iterations = np.arange(1, self.sc_iter+1, 1)

        for x in self.bond_hist_dict:
            hist = self.bond_hist_dict[x]
            s = hist[0] 
            e = hist[1]
            hist = hist[2:]


            ax[0][0].scatter(iterations, np.absolute(hist), s = 4, label=f"bond {s} -> {e}")
            ax[0][1].scatter(iterations, np.angle(hist), s = 4, label=f"bond {s} -> {e}")
        
        for x in self.mu_hist_dict:
            mu_hist = self.mu_hist_dict[x]
            ax[1][0].scatter(iterations, mu_hist, s = 4, label=f"site {x}")


        fig.suptitle(rf"evolution of mean field parameters over iterations for {int((self.T+1)*(self.T+2)/2)} sites ( $\beta$ ={self.beta}, $\kappa$ = {self.kappa})", fontsize = 'x-large')
        for axes in ax.flat:
            axes.set_xlabel("iterations", fontsize = 'large')
        ax[0][0].set_ylabel("|$\chi_{ij}$|", fontsize = 'x-large')

        ax[0][1].set_ylabel("phase of $\chi_{ij}$", fontsize = 'x-large')
        ax[0][1].set_ylim(-np.pi, np.pi)

        ax[1][0].set_ylabel("$\mu_i$", fontsize = 'x-large')


        ax[0][0].legend()
        ax[0][1].legend()
        ax[1][0].legend()
  

    def real_space_plot(self):
            
        c_max = self.T+1
        grid = [[-self.T/2,-np.sqrt(3)*self.T/4]] # initialize with one site 

        for i in np.arange(2, int((self.T+1)*(self.T+2)/2)+1): # site 1 at (0,0) is already included!
            c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))) # column number c of site i
            p = i - int(c*(c-1)/2)  # position p in column c, counting starts at 0
            cords = [[(c-1)-(p-1)/2 - self.T/2, np.sqrt(3)*(p-1)/2 - np.sqrt(3)*self.T/4]] # calculate coordinates s.t. the origin is in the middle of the triangle
            grid = np.append(grid, cords,  axis = 0)

        fig, ax = plt.subplots()
            
        ax.scatter(grid[:,0], grid[:,1], c = 'black', s = 20)

                








