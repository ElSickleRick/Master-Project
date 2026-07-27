import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as ptch
import matplotlib.colors as clr
import matplotlib.cm as cm

class pre_analysis:

    def __init__(self, T, link_dict, plaqu_dict, pre_ana_paras):
        
        self.T = T
        self.link_dict = link_dict
        self.plaqu_dict = plaqu_dict
        self.mu_length, self.bond_length, self.plaqu_length = pre_ana_paras

        return

    def hist_dict_gen(self):

        if self.mu_length > int((self.T+1)*(self.T+2)/2): 
            self.mu_length = int((self.T+1)*(self.T+2)/2)
        mu_select = np.random.choice(np.arange(0, int((self.T+1)*(self.T+2)/2)), self.mu_length, replace = False) # select mu_s to plot
        mu_hist_dict =  {}
        for i in mu_select:
            mu_hist_dict.update({i: []})
        
        if self.bond_length > int(3*(self.T)*(self.T+1)/2): 
            self.bond_length = int(3*(self.T)*(self.T+1)/2)
        bond_select = np.random.choice(list(self.link_dict.keys()), self.bond_length, replace = False) # select bonds to plot
        bond_hist_dict = {}
        for i in bond_select:
            bond_hist_dict.update({i: [self.link_dict[i][0], self.link_dict[i][1]]})
        
        if self.plaqu_length > int(self.T**2): 
            self.plaqu_length = int(self.T**2)
        plaqu_select = np.random.choice(list(self.plaqu_dict.keys()), self.plaqu_length, replace = False) # select plaquettes to plot
        plaqu_hist_dict = {}
        for i in plaqu_select:
            plaqu_hist_dict.update({i : [self.plaqu_dict[i][0], self.plaqu_dict[i][1], []]})
        
        return mu_hist_dict, bond_hist_dict, plaqu_hist_dict






class post_analysis:

    def __init__(self, T, kappa, beta, C, mag_elas, strain_cord_dict, plaqu_dict, pop_link_dict, eival, eivec, mu_arr):
        
        self.T = T
        self.kappa = kappa
        self.beta = beta
        self.C = C
        self.mag_elas = mag_elas
        self.strain_cord_dict = strain_cord_dict
        self.plaqu_dict = plaqu_dict
        self.pop_link_dict = pop_link_dict 
        self.eival = eival
        self.eivec = eivec
        self.mu_arr = mu_arr
        return

    def plaqu_dict_check(self, target):
    
        """
        checks wether a configuration complies with certain conditions on the plaquette flux pattern

        INPUTS:
        -------
        target: float 
                target flux in every plaquette (more complex conditions are not supported yet)

        OUTPUTS:
        -------
        target_con: Boolean
                True if condition applies, False if not 
        """

        target_con = True
    
        tolerance = 10**(-2) # maximum tolerance for deviations from target flux

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


            if np.absolute(phase - target) > tolerance: 
                target_con = False 
                break 

        return target_con 
    

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

        fig, ax = plt.subplots()
        ax.hist(Chi_abs_arr) 

        return Chi_abs_arr

    def Chi_ph_dist_plot(self):
        
        Chi_ph_arr = []

        for x in self.pop_link_dict:
            
            Chi_ph_arr.append(np.angle(self.pop_link_dict[x][3]))

        fig, ax = plt.subplots()
        ax.hist(Chi_ph_arr)

        return(Chi_ph_arr)

    def mu_dist_plot(self):
        
        fig, ax = plt.subplots()
        ax.hist(self.mu_arr)

        return

    



    def free_en_calc(self):
        '''
        calculates the free energy desnsity of a given mean field solution

        Returns:
        --------
        free_en: float 
                free energy  
        '''


        F = 0
        for x in self.eival:
            if x < 0:
                F += 2*x-(2/self.beta)*np.log(1+np.exp(self.beta*x))
            elif x > 0:
                    F += -(2/self.beta)*np.log(1+np.exp(-self.beta*x))

        for x in self.pop_link_dict:
            s, e, J, Chi = self.pop_link_dict[x]
            F += 2*(J*(1+self.kappa/2)- 6*J*self.kappa*(np.absolute(Chi)**2))*(np.absolute(Chi)**2) 

        mu_part = -np.sum(self.mu_arr)

        return 2*(F + mu_part)/((self.T+1)*(self.T+2))
 
    def free_energy_iter_plot(self, sc_iter, free_energy_hist):

        fig, ax = plt.subplots()
        iterations = np.arange(1, sc_iter+1, 1)

        ax.scatter(iterations, free_energy_hist, s = 4)
        ax.set_yscale('symlog')
        ax.set_title("evolution of 'free energy'")
        ax.set_ylabel("F in units of J")
        ax.set_xlabel("iterations")

        print('free energy: ', free_energy_hist[-1], 'J')
        
        return

    def MF_iter_plot(self, sc_iter, mu_hist_dict, bond_hist_dict, plaqu_hist_dict):

        fig, ax = plt.subplots(2,2)
        iterations = np.arange(1, sc_iter+1, 1)

        for x in bond_hist_dict:
            hist = bond_hist_dict[x]
            s = hist[0] 
            e = hist[1]
            history = hist[2:]


            ax[0][0].scatter(iterations, np.absolute(history), s = 4, label=f"bond {s} -> {e}")
            ax[0][1].scatter(iterations, np.angle(history), s = 4, label=f"bond {s} -> {e}")
        
        for x in mu_hist_dict:
            mu_hist = mu_hist_dict[x]
            ax[1][0].scatter(iterations, mu_hist, s = 4, label=f"site {x+1}")

        for p in plaqu_hist_dict:
            orientation, corners, hist = plaqu_hist_dict[p]

            ax[1][1].scatter(iterations, hist, s = 4, label=f"plaquette {corners} ({orientation})") 



        fig.suptitle(rf"evolution of mean field parameters over iterations for {int((self.T+1)*(self.T+2)/2)} sites ( $\beta$ ={self.beta}, $\kappa$ = {self.kappa})", fontsize = 'x-large')
        for axes in ax.flat:
            axes.set_xlabel("iterations", fontsize = 'large')

        ax[0][0].set_ylabel("|$\chi_{ij}$|", fontsize = 'x-large')

        ax[0][1].set_ylabel("phase of $\chi_{ij}$", fontsize = 'x-large')
        ax[0][1].set_ylim(-np.pi-1/4, np.pi+1/4)
        ax[0][1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],)
        ax[0][1].set_yticklabels([f"-$\pi$", f"$-\pi/2$", "0", f"$\pi/2$", f"$\pi$"])

    

        ax[1][0].set_ylabel("$\mu_i$", fontsize = 'x-large')

        ax[1][1].set_ylabel("plaquette flux")
        ax[1][1].set_ylim(-np.pi-1/4, np.pi+1/4)
        ax[1][1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],)
        ax[1][1].set_yticklabels([f"-$\pi$", f"$-\pi/2$", "0", f"$\pi/2$", f"$\pi$"])


        for axes in ax.flat:
            axes.legend()
            axes.grid()  

    def real_space_plot(self):
        
        variant = "t" # if "chi" -> abs(chi) ist plotted, if "t" -> hopping amplitude is plotted

        lw_min = 0
        lw_max = 12
        t_min = 0
        t_max = 5
        chi_abs_min = 0
        chi_abs_max = 0.5

        c_max = self.T+1
        grid = np.empty((0,2)) # initialize with one site 
        
        fig, ax = plt.subplots()

        for i in range(1, int((self.T+1)*(self.T+2)/2+1)):
            
            cords = np.array([self.strain_cord_dict[i][1]])
            grid = np.append(grid, cords, axis = 0)

                    
        # ax.scatter(grid[:,0], grid[:,1], marker = 'x', c = 'k',  s = 40, zorder=2) 

        for link in self.pop_link_dict:

            s, e, J, chi = self.pop_link_dict[link]

            x = [grid[s-1][0], grid[e-1][0]]
            y = [grid[s-1][1], grid[e-1][1]]
            chi_abs = np.absolute(chi)

            if chi_abs < 0.001:
                ax.plot(x, y, c = 'grey', linestyle = 'dotted', zorder = 3)
            
            else:
                if variant == "chi":
                    lw = (lw_max - lw_min)/(chi_abs_max - chi_abs_min)*(np.absolute(chi)-chi_abs_min) + lw_min

                elif variant == "t":
                    if J*np.absolute(chi)*(1+self.kappa/2-4*self.kappa*np.absolute(chi)**2) > t_max:
                            print("warning: upper t-bound for line thicknes too small")
                    lw = (lw_max - lw_min)/(t_max - t_min)*(J*np.absolute(chi)*(1+self.kappa/2-4*self.kappa*np.absolute(chi)**2-t_min)) + lw_min

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

        ax.scatter(0, 0, s = 8, c = 'w', marker = "+", zorder = 3) # mark center
        sm = cm.ScalarMappable(norm = norm, cmap = cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax = ax) 
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels([f"$-\pi$", f"$-\pi / 2$", "0", f"$\pi / 2$", f"$\pi$"])
        ax.grid()

    def DOS_hist(self): 
        
        fig, ax = plt.subplots()
        ax.hist(self.eival, bins = 24)
        ax.set_title(rf"DOS for {int((self.T+1)*(self.T+2)/2)} sites, $\kappa$ = {self.kappa}, $\beta$ = {self.beta}")

    def Chi_path_plot(self):
        
        mid = int(np.ceil((self.T/2+1))-1)

        path = [np.absolute(self.pop_link_dict[str(int(self.T*(self.T+1)/2)) + str(int((self.T+1)*(self.T+2)/2))][3])]

        for i in np.arange(self.T+1, 1, -1):
            s = int(self.T*(self.T+1)/2+i)
            path.append(np.absolute(self.pop_link_dict[str(int(s-1)) + str(s)][3]))

        for i in np.arange(self.T+1, mid, -1):
            s = int((i-1)*i/2+1)
            sb = int((i-2)*(i-1)/2+1)
            path.append(np.absolute(self.pop_link_dict[str(sb)+str(s)][3]))

        for i in np.arange(1, mid):
            s = int(mid*(mid-1)/2 + i)
            path.append(np.absolute(self.pop_link_dict[str(s)+str(int(s+1))][3]))

        for i in np.arange(mid, self.T+1):
            s = int(i*(i+1)/2)
            sa = int((i+1)*(i+2)/2)
            path.append(np.absolute(self.pop_link_dict[str(s)+str(sa)][3]))
        
        path.append(np.absolute(self.pop_link_dict[str(int(self.T*(self.T+1)/2 + self.T)) + str(int(self.T*(self.T+1)/2+self.T+1))][3])) 
        
        fig, ax = plt.subplots()
        ax.scatter(np.arange(0, len(path)), path)
        
        bl = self.T
        rmid = int(bl-mid)

        ticks =[0.5, mid + 0.5, bl + 0.5, bl + rmid + 0.5, 2*bl + 0.5, 2*bl + mid + 0.5]

        ax.set_xticks( ticks ) 
        ax.set_xticklabels(["C", "M", "C", "M", "M", "C"])
        ax.grid()
            


    

            

        








