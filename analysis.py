import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as ptch
import matplotlib.colors as clr
import matplotlib.cm as cm

class analysis:

    def __init__(self, T ,kappa, beta, plaqu_dict, pop_link_dict, eival, eivec, mu_hist_dict, mu_arr, bond_hist_dict, sc_iter):
        
        self.T = T
        self.kappa = kappa
        self.beta = beta
        self.plaqu_dict = plaqu_dict
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
            F += (J*(1+self.kappa/2)- (3/2)*J*self.kappa*(np.absolute(Chi))**2)*(np.absolute(Chi))**2 
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

        ax[0][0].grid()
        ax[0][1].grid()
        ax[1][0].grid()
  

    def real_space_plot(self):
            
        c_max = self.T+1
        grid = [[-self.T/2,-np.sqrt(3)*self.T/4]] # initialize with one site 
        
        fig, ax = plt.subplots()

        for i in np.arange(2, int((self.T+1)*(self.T+2)/2)+1): # site 1 at (0,0) is already included!
            c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))) # column number c of site i
            p = i - int(c*(c-1)/2)  # position p in column c, counting starts at 0
            cords = [[(c-1)-(p-1)/2 - self.T/2, np.sqrt(3)*(p-1)/2 - np.sqrt(3)*self.T/4]] # calculate coordinates s.t. the origin is in the middle of the triangle
            grid = np.append(grid, cords,  axis = 0)
                    
        # ax.scatter(grid[:,0], grid[:,1], marker = 'x', c = 'k',  s = 40, zorder=2) 

        for link in self.pop_link_dict:
            
            lw_min = 0
            lw_max = 8
            chi_abs_min = 0
            chi_abs_max = 0.5

            s, e, J, chi = self.pop_link_dict[link]

            x = [grid[s-1][0], grid[e-1][0]]
            y = [grid[s-1][1], grid[e-1][1]]
            chi_abs = np.absolute(chi)

            if chi_abs < 0.001:
                ax.plot(x, y, c = 'grey', linestyle = 'dotted', zorder = 3)
            
            else:
                lw = (lw_max - lw_min)/(chi_abs_max - chi_abs_min)*(np.absolute(chi)-chi_abs_min) + lw_min      
                ax.plot(x, y, c = 'k', linewidth = lw, zorder = 3)
        
        cmap = clr.LinearSegmentedColormap.from_list("periodic", ["purple", "blue", "white",  "red", "purple"])
        norm = clr.Normalize(vmin = -np.pi, vmax = np.pi)

        for plaqu in self.plaqu_dict:
            orientation, corners = self.plaqu_dict[plaqu]
            

            if orientation == 'up':
                base = self.pop_link_dict[str(corners[0]) + str(corners[1])][3]
                right = self.pop_link_dict[str(corners[1])+str(corners[2])][3]
                left = self.pop_link_dict[str(corners[0]) + str(corners[2])][3]
                phase = np.angle(base*np.conjugate(left)*right) 


            elif orientation == 'down':
                right = self.pop_link_dict[str(corners[0]) + str(corners[2])][3]
                top = self.pop_link_dict[str(corners[1]) + str(corners[2])][3]
                left = self.pop_link_dict[str(corners[0]) + str(corners[1])][3]
                phase = np.angle(right*np.conjugate(top)*np.conjugate(left))
            
            color = cmap(norm(phase))
            triangle = ptch.Polygon([grid[int(corners[0]-1)], grid[int(corners[1]-1)], grid[int(corners[2]-1)]], color = color, zorder = 1)
            ax.add_patch(triangle)

        sm = cm.ScalarMappable(norm = norm, cmap = cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax = ax) 
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels([f"$-\pi$", f"$-\pi / 2$", "0", f"$\pi / 2$", f"$\pi$"])

    def DOS_hist(self): 
        
        fig, ax = plt.subplots()
        ax.hist(self.eival, bins = 50)


    

            

        








