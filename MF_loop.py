import numpy as np
import scipy.linalg as la 
from multiprocessing import pool

class MF_loop:


    def __init__(self, pop_link_dict,  beta, eival, eivec,  N_diff_bounid, mu_step,  Chi_abs_dif_bound, Chi_ph_dif_bd,  Chi_step):
	# mu step probably has to be adjusetd!
	
	self.pop_link_dict = pop_link_dict
        self.beta = beta 
	self.eival = eival
	self.eivec = eivec 
        self.N_dif_bound = N_dif_bound
	self.mu_step = mu_step
        self.Chi_abs_dif_bound = Chi_abs_dif_bound
	self.Chi_ph_dif_bd = Chi_ph_dif_bd
        self.Chi_step = Chi_step
        return


    def thermal_calc(self):
        '''
        Calculates the Boltzman weights as well as the partition fucntion

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
    

    def N_calc(self, Z, weights):

        '''
        Calculates the expectation value of N on all sites 

        Parameters:
        ---------
        Z: float
                partition function

        weights: array of floats 
                thermal (Boltzman) weights ordered accoarding to the energies (ascending)

        Returns:
        ------- 
        N_arr: array of floats
                expectation values of particle numbers on every site, ordered in the usual way
        '''

        return (2/Z)*(np.absolute(self.eivec) @ weights)
    

    def Chi_calc(self, s, e, weights):  # keep in mind: numpy is row first
        
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
     
        return weights @ (self.eivec[s-1][:]*np.conjugate(self.eivec[e-1][:]))


    def Chi_comp(self,s,e,Chi_new):

	a = pop_link_dict[str(s) + str(e)][2]
	b = pop_link_dict[str(s) + str(e)][3]

	if dtype(a) == np.complex128:
		Chi_old = a
	
	elif dtype(b) == np.complex128:
		Chi_old = b

	else: 
		print('error: neither Chi is not  complex')
	
	if np.absolute(np.absolute(Chi_old) -np.absolute(Chi_new))  < self.Chi_abs_dif_bd and np.absulte(np.phase(Chi_old) - np.phase(Chi_new)) < Chi_ph_dif_bd:
		return Chi_old, True

	else: 
		
	 	

    def mu_comp(self):

       
        if all(np.absolute(N_arr - 1) < self.N_dif_bound) == True:

            return mu_arr, True

        else: 

            return mu_arr + 0.1*mu_step*np.random.rand*(N_arr - 1), False # Ill probably have to adjust the factor 0.1 later to fit the local energy scale	


 
        



        


    
    
