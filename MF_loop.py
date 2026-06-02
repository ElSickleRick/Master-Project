import numpy as np
import scipy.linalg as la 
from multiprocessing import pool

class MF_loop:


    def __init__(self, T, beta, eival, eivec, N_dif_bd, mu_hist_dict, mu_step, pop_link_dict, bond_hist_dict, Chi_dif_bd, rm_scale, plaqu_hist_dict,  conv, rng):
	# mu step probably has to be adjusetd!
        	
        self.T = T
        self.beta = beta 
        self.eival = eival
        self.eivec = eivec 
        self.N_dif_bd = N_dif_bd
        self.mu_hist_dict = mu_hist_dict
        self.mu_step = mu_step
        self.pop_link_dict = pop_link_dict
        self.bond_hist_dict = bond_hist_dict
        self.Chi_dif_bd = Chi_dif_bd
        self.rm_scale = rm_scale
        self.plaqu_hist_dict = plaqu_hist_dict
        self.conv = conv
        self.rng = rng

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
    
    def update(self, mu_old, fe_di):
       
        conv = True 
        
        fe_di_mat = np.zeros((int((self.T+1)*(self.T+2)/2), int((self.T+1)*(self.T+2)/2)))
        np.fill_diagonal(fe_di_mat, 2*fe_di)

        new = ((self.eivec @ fe_di_mat) @ np.transpose(np.conjugate(self.eivec)))

        N_arr = np.diag(new).astype(float)
        mu_new = mu_old + self.mu_step*self.rng.random(int((self.T+1)*(self.T+2)/2))*(N_arr - 1)


        if all(np.absolute(N_arr - 1) < self.N_dif_bd) != True:
            conv = False
       
        bond_hist_dict_keys = self.bond_hist_dict.keys()

        for x in self.pop_link_dict:

            s, e, J, Chi_old = self.pop_link_dict[x]
            alpha = self.rm_scale*self.rng.random()

            Chi_new= new[e-1][s-1]
            Chi_update = (1-alpha)*Chi_old + alpha*Chi_new

            self.pop_link_dict[str(s)+str(e)][3] = Chi_update
            
            if (np.absolute(np.imag(Chi_old - Chi_new))  > self.Chi_dif_bd) or (np.absolute(np.real(Chi_old - Chi_new)) > self.Chi_dif_bd):

                conv = False


        for i in self.mu_hist_dict:
            self.mu_hist_dict[i].append(mu_new[i])

        for x in self.bond_hist_dict:
            self.bond_hist_dict[x].append(self.pop_link_dict[x][3])

        for p in self.plaqu_hist_dict:
            orientation, corners = self.plaqu_hist_dict[p][:2]

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
        
            self.plaqu_hist_dict[p][2].append([phase])


        return mu_new, conv


            


        



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
			

 
        



        


    
    
