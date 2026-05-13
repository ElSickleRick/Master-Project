import numpy as np
import scipy.linalg as la 
from multiprocessing import pool

class MF_loop:


    def __init__(self, beta, eival, eivec,  N_dif_bd, mu_step, pop_link_dict, Chi_dif_bd):
	# mu step probably has to be adjusetd!
        	
        self.pop_link_dict = pop_link_dict
        self.beta = beta 
        self.eival = eival
        self.eivec = eivec 
        self.N_dif_bd = N_dif_bd
        self.mu_step = mu_step
        self.Chi_dif_bd = Chi_dif_bd
        return


    def lin_fe_di(self, Ebeta):

        '''
        linearization of fermi dirac distribution 
        '''

        if Ebeta < -2:
            return 1
         
        elif Ebeta > 2:
            return 0

        else:
            return (1/4)*(2-Ebeta)
    
    def update(self, mu_arr, fedi):
        
        conv = True
        new = (fedi*np.conjugate(self.eivec)) @ self.eivec

        N_arr = np.diag(new)
        print("N array is", N_arr)
        mu_arr += self.mu_step*np.random.rand()*(N_arr - 1) # Ill probably have to adjust the factor 0.1 later to fit the local energy scale 

        if all(np.absolute(N_arr - 1) < self.N_dif_bd) != True:
            conv = False

        for x in self.pop_link_dict:

            s, e, J, Chi_old = self.pop_link_dict[link]
            alpha = 0.4*np.random.rand()
       
            Chi_new= 2*((fe_di*np.conjugate(self.eivec[s-1][:]))@self.eivec[:][e-1])
            self.pop_link_dict[str(s)+str(e)][3] = (1-alpha)*Chi_old + alpha*Chi_new

            if (np.absolute(np.imag(Chi_old - Chi_new))  > self.Chi_dif_bd) or (np.absolute(np.real(Chi_old - Chi_new)) > self.Chi_dif_bd):

                conv = False

        return mu_arr 


            


        



    def mu_update(self, mu_arr, fe_di):

        '''
        Calculates the expectation value of N on all sites 

        Parameters:
        ---------
        mu_arr: array of floats
                array containing all local chemical potentials in the usual order
        Z: float
                partition function

        fe_di: array of floats 
                values of fermi-dirac distribiution for each eigenenergies (ascending order)

        Returns:
        ------- 
        N_arr: array of floats
                expectation values of particle numbers on every site, ordered in the usual way
        '''

        N_arr = 2*(np.absolute(self.eivec)**2 @ fe_di) # I am verry happy that this works but I should check it again
        mu_arr += self.mu_step*np.random.rand()*(N_arr - 1) # Ill probably have to adjust the factor 0.1 later to fit the local energy scale    
        

        if all(np.absolute(N_arr - 1) < self.N_dif_bd) == True:
            return True, mu_arr

        else: 
            return False, mu_arr
    

    def Chi_update(self, link, fe_di):  # keep in mind: numpy is row first
        
        '''
        Calculates the expectation value of Chi on the link s s -> e.
            -> always calculates Chi_{se}, such that particles on site e are annihilated and particles on site s are created 

        Parameters:
        ----------
        s: int 
                site where the bond starts (s<e)
                counting convention: start with 1
        e: int 
                site where the bond ends (s<e)
                counting convention: start with 1
        fedi: array of floats
                values of fermi-dirac distribiution for each eigenenergie (ascending order)

        Returns:
        -------
        conv_check: boolean
                True if updated calc is sufficiently close to old Chi, False otherwise
                
        
        '''
        s, e, J, Chi_old = self.pop_link_dict[link]
        alpha = 0.4*np.random.rand()
       
        Chi_new= 2*((fe_di*np.conjugate(self.eivec[s-1][:]))@self.eivec[:][e-1])
        self.pop_link_dict[str(s)+str(e)][3] = (1-alpha)*Chi_old + alpha*Chi_new	
        
        if (np.absolute(np.imag(Chi_old - Chi_new))  < self.Chi_dif_bd) and (np.absolute(np.real(Chi_old - Chi_new)) < self.Chi_dif_bd):
             return True

        else:
            return False 
			

 
        



        


    
    
