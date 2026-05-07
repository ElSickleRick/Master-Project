import numpy as np
import scipy.linalg as la 
from multiprocessing import pool

class MF_loop:

    def __init__(self, beta, N_diff_bound, mu_step, Chi_diff_bound, Chi_step):
        # possible pass on parameters: beta, N_diff_bound, mu_step, Chi_diff_bound, Chi_step
        # maybe pass on eival and eivec as well?
        self.beta = beta 
        self.N_diff_bound = N_diff_bound 
        self.mu_step = mu_step
        self.Chi_diff_bound = Chi_diff_bound
        self.Chi_step = Chi_step
        return 

    def Ham_diag(Ham): #merge with thermal_calc?

        eival, eivec = la.eigh(Ham, lower=False)

        return eival, eivec  

    def thermal_calc(self, eival):
        '''
        Calculates the Boltzman weights as well as the partition fucntion

        Parameters:
        ----------
        eival: array of floats 
                eigenvalues of Hamiltonains, ascendingly ordered

        Returns:
        -------
        weights: array of floats
                thermal (Boltzman) weights ordered accorading the energies (ascending)
        Z: float
                partition function
        '''
        weights = np.exp(-self.beta*eival)

        return weights, np.sum(weights)
    
    def N_calc(Z, eivec, weights):

        '''
        Calculates the expectation value of N on all sites 

        Parameters:
        ---------
        Z: float
                partition function
        eivec: 2D array of floats
                matrix of eigenvectors of Hamiltonian (eigenvectors are columns)
        weights: array of floats 
                thermal (Boltzman) weights ordered accoarding to the energies (ascending)

        Returns:
        ------- 
        N_arr: array of floats
                expectation values of particle numbers on every site, ordered in the usual way
        '''

        return (2/Z)*(np.absolute(eivec) @ weights)
    
    def Chi_calc(s, e, weights, eivec):  # keep in mind: numpy is row first
        
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
        eivec: 2D array of floats
                matrix of eigenvectors of Hamiltonian (eigenvectors are columns)

        Returns:
        -------
        Chi_new: np.complex128
                calculated Chi on the s -> e bond
                
        
        '''

        
        return weights @ (eivec[s-1][:]*np.conjugate(eivec[e-1][:]))


    def mu_comp(self):

       
        if all(np.absolute(N_arr - 1) < self.N_diff_bound) == True:

            return mu_arr, True

        else: 

            return mu_arr + self.mu_step*(N_arr - 1), False 
        



        


    
    
