import numpy as np
import scipy.linalg as la 
from multiprocessing import pool

from system_init import system_init 
from analysis import pre_analysis, post_analysis

class methods:


    def __init__(self, T, beta, kappa, C, mag_elas):
        """
        PARAMETERS
        ----------
        T: int
            number of triangles on the basis of the sample
        beta: float
            inverse temperature
        kappa: float
            biquadratic exchange coupling constant
        """

        self.T = T
        self.beta = beta 
        self.kappa = kappa
        self.C = C
        self.mag_elas = mag_elas
        self.N = int((T+1)*(T+2)/2) # # sites
    
        return

    def free_en_calc(self, eival, mu_arr, pop_link_dict):
        '''
        calculates the free energy desnsity of a given mean field solution

        Returns:
        --------
        free_en: float 
                free energy  
        '''


        F = 0
        for x in eival:
            if x < 0:
                F += 2*x-(2/self.beta)*np.log(1+np.exp(self.beta*x))
            elif x > 0:
                    F += -(2/self.beta)*np.log(1+np.exp(-self.beta*x))

        for x in pop_link_dict:
            s, e, J, Chi = pop_link_dict[x]
            F += 2*(J*(1+self.kappa/2)- 6*J*self.kappa*(np.absolute(Chi)**2))*(np.absolute(Chi)**2) 

        mu_part = -np.sum(mu_arr)

        return 2*(F + mu_part)/((self.T+1)*(self.T+2))

    def lin_fe_di(self, E):

        '''
        returns value of linearized fermi-dirac distribution at a given energy value -> all states are by default doubly degenerate
        '''

        if E*self.beta < -2: 
            return 2
         
        elif E*self.beta > 2:
            return 0

        else: # linear region around chemical potential
            return (1/2)*(2-E*self.beta)

    
    def MF_update(self, rng, convergence_paras, eival, eivec, pop_link_dict, mu_old):

        """
        updates the mean field parameters (bond parameters + chemical potentials)

        PARAMETERS
        ----------
        rng: generator object
            PCG64 random number generator (in a certain state)
        convergence_paras: array, size 5
            parameters controlling convergence behaviour (see main)
        eival: y, size N
            eigenvalues in ascending order
        eivec: 2D-array, size [N,N]
            eigenvectors, eivec[:,i] corresponds to eival[i]
        pop_link_dict: dictionary
            keys are bonds, values are [start, end, exc-coupling, bond-parameter]
        mu_old: array, size N
            chemical potentials before the update

        RETURNS
        -------
        mu_new: array
            updated chemical potentials
        conv: boolean
            True if convergence occured, False if not
        """
       
        mu_step_base, mu_rm_scale, chi_rm_scale, N_dif_bd, chi_dif_bd = convergence_paras 

        conv = True 
        
        fe_di = [self.lin_fe_di(x) for x in eival] # caclulate fermi-dirac distributions 

        new = ((eivec*fe_di) @ np.transpose(np.conjugate(eivec))) # calculate particle numbers (diagonal) and bond-parameters (off-diagonal)

        N_arr = np.diag(new).astype(float) # extract particle number
        mu_new = mu_old + (np.full(self.N, mu_step_base) + mu_rm_scale*(2*rng.random(self.N)-1))*(N_arr - 1) # update chemcial potentials


        if all(np.absolute(N_arr - 1) < N_dif_bd) != True: # check if particle number is close enough to one
            conv = False
       

        # below normal verison
        for x in pop_link_dict:

            s, e, J, chi_old = pop_link_dict[x]
            alpha = chi_rm_scale*rng.random() # para for random mixing

            chi_new= (1/2)*new[e-1][s-1] # extract bond para (factor 1/2 is convention)
            chi_update = (1-alpha)*chi_old + alpha*chi_new # update bond para in direction of chi_new

            pop_link_dict[str(s)+str(e)][3] = chi_update # write updated bond para to dicitionary 
            
            if (np.absolute(np.imag(chi_old - chi_new))  > chi_dif_bd) or (np.absolute(np.real(chi_old - chi_new)) > chi_dif_bd): # check if chi_new is close enough to chi_old

                conv = False
        
        # below version for uniform bond parameters:
        """
        chi_arr = []
        for x in pop_link_dict: # collect all bond paras
             chi_arr.append(pop_link_dict[x][3])

        chi_mean = np.mean(chi_arr) # calculate mean of bond paras 

        for x in pop_link_dict:
            alpha = chi_rm_scale*rng.random() # para for random mixing 
            pop_link_dict[x][3] = (1-alpha)*pop_link_dict[x][3] +  alpha*chi_mean # update bond para in direction of chi_new
        """

        return mu_new, conv


    def MF_solver(self, rng, iter_paras, pre_ana_paras, convergence_paras, link_dict, plaqu_dict, mu_arr, pop_link_dict):

        """
        calculates mean field solution for a system initialization

        PARAMETERS
        ----------
        rng: generator object
            PCG64 random number generator (in a certain state)
        iter_parars: array, size 2
            paras for possbile premature termination of algorithm (see main)
        pre_ana_paras: array, size 3
            paras for tracking convergence (see main)
        convergence_paras: array, size 5
            parameters controlling convergence behaviour (see main)
        link_dict: dicitionary
            keys are bonds, values are [start, end]
        plaqu_dict: dictionary 
            keys are plaquettes, values are [orientation, [corner1, corner2, corner3]]
        mu_arr: array, size N
            chemcial potentials
        pop_link_dict: dictionary
            keys are bonds, values are [start, end, exc-coupling, bond-paras

        RETURNS
        -------
        mu_arr: array, size N
            chemcial potentials
        pop_link_dict: dictionary 
            keys are bonds, values are [start, end, exc-coupling, bond-paras]
        mu_hist_dict: dictionary
            tracks evolution of chemical potentials
        bond_hist_dict: dictionary 
            tracks evolution of bond parameters
        plaqu_hist_dict: dictionary 
            tracks evolution of plaquette fluxes
        free_energy_hist: array
            tracks evolution of free energy
        eival: array, size N
            eigenvalues in ascending order
        eivec: 2D array, size [N,N]
            eigenvectors, eivec[:,i] corresponds to eival[i]
        sc_iter: int
            # iteration
        """
        
        max_iter_cond, sc_iter_max = iter_paras  

        free_en_hist = [] # array for tracking evolution of "free energy"

        mu_length, bond_length, plaqu_length = pre_ana_paras 
        pre_ana = pre_analysis(self.T, link_dict, plaqu_dict, pre_ana_paras)
        mu_hist_dict, bond_hist_dict, plaqu_hist_dict = pre_ana.hist_dict_gen() # randomly choose which chem potentials, bond paras and plaquettes to track

        conv = False 
        interrupt = False 
        sc_iter = 0 # number of iterations of self-consistency loop 

        while conv == False:
    
            conv = True
            
            init = system_init(self.T, self.kappa, rng, self.C, self.mag_elas)
            Ham = init.Ham_builder(pop_link_dict, mu_arr) # build Hamiltonian
            eival, eivec = la.eigh(Ham, lower = False) # daigonalize Hamiltonian, entries are in upper traingle! 

            mu_arr, conv  = self.MF_update(rng, convergence_paras, eival, eivec, pop_link_dict, mu_arr) # update chem potentials and bond paras 

            sc_iter += 1       

            for i in mu_hist_dict: # chem potential tracking
                mu_hist_dict[i].append(mu_arr[i])

            for x in bond_hist_dict: # bond para tracking
                bond_hist_dict[x].append(pop_link_dict[x][3])

            for p in plaqu_hist_dict: # plaquette tracking
                orientation, corners = plaqu_hist_dict[p][:2]

                if orientation == 'up':
                    base = pop_link_dict[str(corners[0]) + str(corners[1])][3]
                    right = pop_link_dict[str(corners[1])+str(corners[2])][3]
                    left = pop_link_dict[str(corners[0]) + str(corners[2])][3]
                    phase = np.angle(base*np.conjugate(left)*right) 


                elif orientation == 'down':
                    right = pop_link_dict[str(corners[0]) + str(corners[2])][3]
                    top = pop_link_dict[str(corners[1]) + str(corners[2])][3]
                    left = pop_link_dict[str(corners[0]) + str(corners[1])][3]
                    phase = np.angle(right*np.conjugate(top)*np.conjugate(left))
        
                plaqu_hist_dict[p][2].append([phase]) 

            f = self.free_en_calc(eival, mu_arr, pop_link_dict) # calculate "free energy" of cinfiguration
            free_en_hist.append([f])

            if sc_iter >= sc_iter_max and max_iter_cond == True: # (conditional) premature temrination of algorithm 
                print("self consistency loop intrrupted: to many iterations (",sc_iter_max,")")
                interrupt = True 
                conv = True 
                break

            if (sc_iter-1) % 500 == 0 and sc_iter != 1:
                print("finished iteration", sc_iter-1)
        

        if interrupt == False:
            print("self-consistency loop finished normally after", sc_iter, "iterations")

        return(mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, free_en_hist, eival, eivec, sc_iter)

            
       
    def random_phase_var(self, rng, pop_link_dict):

        """
        randomly varies phase of all bond parameters
        """

        for x in pop_link_dict:

            pop_link_dict[x][3] *= np.exp(1j*np.pi*0.1*(rng.random()*2 - 1))

        return pop_link_dict


    def gauge_trafo(self, rng,  pop_link_dict):

        """
        performs gauge transformation
        """


        local_trafo = [0] # array for saving the local transformations (phase only). 0 is filler since site counting starts at 1. convention: c_i^dagger -> exp(i*phase)*c_i^\dagger

        trafo = "random"

        if trafo == "stripes":

            for c in np.arange(1, T+2): # every other column gets the phase pi/7 or 2*pi/7
        
                if int(c % 2) == 1: 
                    local_trafo.extend((np.pi/7)*np.ones(c))
        
                elif int(c % 2) == 0:
                    local_trafo.extend((2*np.pi/7)*np.ones(c))

        elif trafo == "random":

            local_trafo.extend((np.pi/4)*rng.random(int((self.T+1)*(self.T+2)/2)))



        c_max = self.T + 1

        for i in np.arange(1, int((self.T+1)*(self.T+2)/2 + 1)):

            c = int(np.ceil(-1/2 + np.sqrt(-3/4 + 2*i)))

            if i != c*(c+1)/2: # neighbor above
                pop_link_dict[str(i) + str(int(i+1))][3] *= np.exp(1j*(local_trafo[int(i+1)] - local_trafo[i]))

            if c != c_max: # neighbor above + right and below + right
                pop_link_dict[str(i) + str(int(i+c))][3] *= np.exp(1j*(local_trafo[int(i+c)]-local_trafo[i]))
                pop_link_dict[str(i) + str(int(i+c+1))][3] *= np.exp(1j*(local_trafo[int(i+c+1)] - local_trafo[i]))
        
        return pop_link_dict



