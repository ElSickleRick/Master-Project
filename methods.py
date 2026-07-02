import numpy as np
import scipy.linalg as la 
from multiprocessing import pool

from system_init import system_init 
from analysis import pre_analysis, post_analysis

class methods:


    def __init__(self, T, beta, kappa):

        self.T = T
        self.beta = beta 
        self.kappa = kappa
    
        return


    def lin_fe_di(self, E):

        '''
        linearization of fermi dirac distribution for the case that every energy is by default doubly degenerate (as it is in this case)
        '''

        if E*self.beta < -2:
            return 2
         
        elif E*self.beta > 2:
            return 0

        else:
            return (1/2)*(2-E*self.beta)

    
    def MF_update(self, rng, convergence_paras, eival, eivec, pop_link_dict, bond_hist_dict, mu_old, mu_hist_dict, plaqu_hist_dict):
       
        mu_step_base, mu_rm_scale, chi_rm_scale, N_dif_bd, chi_dif_bd = convergence_paras 

        conv = True 
        
        fe_di = np.array([self.lin_fe_di(x) for x in eival])
        fe_di_mat = np.zeros((int((self.T+1)*(self.T+2)/2), int((self.T+1)*(self.T+2)/2)))
        np.fill_diagonal(fe_di_mat, fe_di)

        new = ((eivec @ fe_di_mat) @ np.transpose(np.conjugate(eivec)))

        N_arr = np.diag(new).astype(float)
        mu_new = mu_old + (mu_step_base*np.ones(int((self.T+1)*(self.T+2)/2)) + mu_rm_scale*(2*rng.random(int((self.T+1)*(self.T+2)/2))-1))*(N_arr - 1)


        if all(np.absolute(N_arr - 1) < N_dif_bd) != True:
            conv = False
       
        bond_hist_dict_keys = bond_hist_dict.keys()
        """
        for x in pop_link_dict:

            s, e, J, Chi_old = pop_link_dict[x]
            alpha = chi_rm_scale*rng.random()

            Chi_new= (1/2)*new[e-1][s-1]
            Chi_update = (1-alpha)*Chi_old + alpha*Chi_new

            pop_link_dict[str(s)+str(e)][3] = Chi_update
            
            if (np.absolute(np.imag(Chi_old - Chi_new))  > chi_dif_bd) or (np.absolute(np.real(Chi_old - Chi_new)) > chi_dif_bd):

                conv = False
        """
        chi_arr = []
        for x in pop_link_dict:
            s, e, J, chi_old = pop_link_dict[x]
            chi_arr.append(chi_old)

        chi_mean = np.mean(chi_arr)

        for x in pop_link_dict:
            alpha = chi_rm_scale*rng.random()
            pop_link_dict[x][3] = (1-alpha)*pop_link_dict[x][3] +  alpha*chi_mean


        for i in mu_hist_dict:
            mu_hist_dict[i].append(mu_new[i])

        for x in bond_hist_dict:
            bond_hist_dict[x].append(pop_link_dict[x][3])

        for p in plaqu_hist_dict:
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


        return mu_new, conv


    def MF_solver(self, rng, iter_paras, pre_ana_paras, convergence_paras, link_dict, plaqu_dict, mu_arr, pop_link_dict):
        
        max_iter_cond, sc_iter_max = iter_paras  

        free_en_hist = []

        mu_length, bond_length, plaqu_length = pre_ana_paras
        pre_ana = pre_analysis(self.T, link_dict, plaqu_dict, pre_ana_paras)
        mu_hist_dict, bond_hist_dict, plaqu_hist_dict = pre_ana.hist_dict_gen()

        conv = False 
        interrupt = False 
        sc_iter = 0 # number of iterations of self-consistency loop 

        while conv == False:
    
            conv = True
            
            init = system_init(self.T, self.kappa, rng)
            Ham = init.Ham_builder(pop_link_dict, mu_arr)
            eival, eivec = la.eigh(Ham, lower = False) # daigonalize Hamiltonian, entries are in upper traingle! 

            mu_arr, conv  = self.MF_update(rng, convergence_paras, eival, eivec, pop_link_dict, bond_hist_dict, mu_arr, mu_hist_dict, plaqu_hist_dict)

            sc_iter += 1
        
            post_ana = post_analysis(self.T, self.kappa, self.beta, plaqu_dict, pop_link_dict, eival, eivec, mu_hist_dict,  mu_arr, bond_hist_dict, plaqu_hist_dict, sc_iter)
            f = post_ana.free_en_calc()
            free_en_hist.append([f])

            if sc_iter >= sc_iter_max and max_iter_cond == True:
                print("self consistency loop intrrupted: to many iterations (",sc_iter_max,")")
                interrupt = True 
                conv = True 
                break

            if (sc_iter-1) % 500 == 0 and sc_iter != 1:
                print("finished iteration", sc_iter-1)
        

        if interrupt == False:
            print("self-consistency loop finished normally after", sc_iter, "iterations")

    
        return(mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, free_en_hist, eival, eivec, sc_iter)

            
       
    def random_phase_var(self, rng):

        for x in self.pop_link_dict:

            self.pop_link_dict[x][3] *= np.exp(1j*np.pi*0.001*(rng.random()*2 - 1))

        return pop_link_dict


    def gauge_trafo(self, pop_link_dict):


        local_trafo = [0] # array for saving the local transformations (phase only). 0 is filler since site counting starts at 1. convention: c_i^dagger -> exp(i*phase)*c_i^\dagger

        trafo = "random"

        if trafo == "stripes":

            for c in np.arange(1, T+2): # every other column gets the phase pi/7 or 2*pi/7
        
                if int(c % 2) == 1: 
                    local_trafo.extend((np.pi/3)*np.ones(c))
        
                elif int(c % 2) == 0:
                    local_trafo.extend((2*np.pi/3)*np.ones(c))

        elif trafo == "random":

            local_trafo.extend((np.pi/7)*rng.random(int((self.T+1)*(self.T+2)/2)))



        c_max = self.T + 1

        for i in np.arange(1, int((self.T+1)*(self.T+2)/2 + 1)):

            c = int(np.ceil(-1/2 + np.sqrt(-3/4 + 2*i)))

            if i != c*(c+1)/2: # neighbor above
                pop_link_dict[str(i) + str(int(i+1))][3] *= np.exp(1j*(local_trafo[int(i+1)] - local_trafo[i]))

            if c != c_max: # neighbor above + right and below + right
                pop_link_dict[str(i) + str(int(i+c))][3] *= np.exp(1j*(local_trafo[int(i+c)]-local_trafo[i]))
                pop_link_dict[str(i) + str(int(i+c+1))][3] *= np.exp(1j*(local_trafo[int(i+c+1)] - local_trafo[i]))
        
        return pop_link_dict



