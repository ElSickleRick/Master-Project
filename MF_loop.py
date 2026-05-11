import numpy as np
import scipy.linalg as la 
from multiprocessing import pool

class MF_loop:


    def __init__(self, beta, eival, eivec,  N_dif_bd, mu_step, pop_link_dict, Chi_abs_dif_bd, Chi_ph_dif_bd):
	# mu step probably has to be adjusetd!
        	
        self.pop_link_dict = pop_link_dict
        self.beta = beta 
        self.eival = eival
        self.eivec = eivec 
        self.N_dif_bd = N_dif_bd
        self.mu_step = mu_step
        self.Chi_abs_dif_bd = Chi_abs_dif_bd
        self.Chi_ph_dif_bd = Chi_ph_dif_bd
        return


    def thermal_calc(self):
        '''
        Calculates the Boltzman weights as well as the partition function

        Parameters:
        ----------
        Returns:
        -------
        weights: array of floats
                thermal (Boltzman) weights ordered accorading the energies (ascending)
        Z: float
                partition function
        '''
        weights = np.exp(-self.beta*self.eival)
      

        return weights, np.sum(weights)
    

    def mu_update(self, mu_arr, Z, weights):

        '''
        Calculates the expectation value of N on all sites 

        Parameters:
        ---------
        mu_arr: array of floats
                array containing all local chemical potentials in the usual order
        Z: float
                partition function

        weights: array of floats 
                thermal (Boltzman) weights ordered accoarding to the energies (ascending)

        Returns:
        ------- 
        N_arr: array of floats
                expectation values of particle numbers on every site, ordered in the usual way
        '''

        N_arr =  (2/Z)*np.matmul(np.absolute(self.eivec)**2, np.conjugate(weights))

        print("N array is" , N_arr)
        print(" sum over N array is", np.sum(N_arr))

        mu_arr += self.mu_step*np.random.rand()*(N_arr - 1) # Ill probably have to adjust the factor 0.1 later to fit the local energy scale       

        if all(np.absolute(N_arr - 1) < self.N_dif_bd) == True:
            return True, mu_arr

        else: 
            return False, mu_arr
    

    def Chi_update(self, link, weights, Z):  # keep in mind: numpy is row first
        
        '''
        Calculates the expectation value of Chi on the link s s -> e.

        Parameters:
        ----------
        s: int 
                site where the bond starts (s<e)
                counting convention: start with 1
        e: int 
                site where the bond ends (s<e)
                counting convention: start with 1
        weights: array of floats
                thermal (Boltzman) weights ordered according to the energies (ascending)


        Returns:
        -------
        Chi_new: np.complex128
                calculated Chi on the s -> e bond
                
        
        '''
        s, e, J, Chi_old = self.pop_link_dict[link]
        Chi_new=  (weights @ (self.eivec[s-1][:]*np.conjugate(self.eivec[e-1][:]))) * (2/Z)
	
        if np.absolute(np.absolute(Chi_old) -np.absolute(Chi_new))  < self.Chi_abs_dif_bd and np.absolute(np.angle(Chi_old) - np.angle(Chi_new)) < self.Chi_ph_dif_bd:
             return True

        else:

            alpha = 0.3*np.random.rand()
            self.pop_link_dict[str(s)+str(e)][3] = (1-alpha)*Chi_old + alpha*Chi_new
            return False 
			

 
        



        


    
    
