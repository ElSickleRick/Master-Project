import numpy as np
import time


class system_init:

    def __init__(self,T, kappa, rng): 

        self.T = T
        self.kappa = kappa
        self.rng = rng
        return


    def uNN_dict_gen(self):

        '''
        Only allows unique nearest neighbors (uNN), such that there is no redundancy.
        -> uNN are always above, above + right and below + right sites if existent

        Returns
        -------
        NN_dict : dictionarie of !unique! uNN of all sites 
        -> keys are site numbers
        -> values are lists of uNN sites
        -> counting starts at 1 and goes to (T+1)*(T+2)/2
        
        
        note: the idea is that to construct the Hamiltonian matrix, one can get 
        away with only cosntructing the upper triangle and deduce the 
        rest by hermicity.
        '''
        
        s = int((self.T+1)*(self.T+2)/2) # number of sites
        c_max = self.T + 1 # maximum column number (see below)
        uNN_dict = {1 : [2,3]}
        
        for i in range(2, s+1): 
            
            uNN = [] # save uNN for site i here
            
            c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))) # caclulate column number c of site i 
            
            if i != c*(c+1)/2: # neighbor above
                uNN.extend([int(i+1)])
                
            if c != c_max: # neighbor above + right and below + right 
                uNN.extend([int(i+c), int(i+c+1)])
            
            uNN_dict.update({i : uNN})
            
        return uNN_dict 


    def link_dict_gen(self):

        '''
        gives empty dictionary of all links in a triangle lattice of given size 
        (see below)

        Returns
        -------
        ul_dict : dict 
                contains all links on the lattice 
                keys are links saved as strings of the form "nm", 
                where n < m are the two sites connected by the links
                values are lists of the form [n,m], n < m
        '''
        
        s = int((self.T+1)*(self.T+2)/2) # number of sites
        c_max = self.T + 1 # maximum column number (see below)
        
        ul_dict = {'12' : [1,2],
                '13' : [1,3]}
        
        for i in range(2, s+1): 
            
            c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))) # caclulate column number c of site i 
            
            if i != c*(c+1)/2: # neighbor above
                ul_dict.update({str(i) + str(i+1) : [i, i+1]})
                
            if c != c_max: # neighbor above + right and below + right 
                ul_dict.update({str(i) + str(i+c) : [i, i+c]})
                ul_dict.update({str(i) + str(i+c+1) : [i, i+c+1]})

        return ul_dict


    def plaqu_dict_gen(self):

        plaqu_dict = {}

        c_max = self.T + 1

        for s in np.arange(1, (self.T+1)*(self.T+2)/2+1):
        
            c = int(np.ceil(-1/2 + np.sqrt(-3/4+2*s))) # column of site s
            p = int(s - c*(c-1)/2) # position p inside column c 

            if c != c_max: # check that site is not in last column

                if p == 1: # case for s first site in column

                    plaqu_dict.update({ str(int(s)) + str(int(s+c)) + str(int(s+c+1)) : ['up', [int(s), int(s+c), int(s+c+1)]] }) # up triangle with s as bottom left corner

                else: # case for site not first in column

                    plaqu_dict.update({ str(int(s)) + str(int(s+c)) + str(int(s+c+1)): ['up', [int(s), int(s+c), int(s+c+1)]] }) # up triangle with s as bottom left corner
                    plaqu_dict.update({ str(int(s-1)) + str(int(s)) + str(int(s+c)): ['down', [int(s-1), int(s), int(s+c)]] }) # down triangle with s as top left corner

        return plaqu_dict



    def J_init(self, link_dict):

        '''
        Placeholder: inserts J's (uniform, equal to 1) into the link dictionary 

        '''

        J = np.ones(int(3*self.T*(self.T+1)/2))
        # J = 1.75*self.rng.random(int(3*self.T*(self.T+1)/2)) + 0.25
        i = 0
    
        
        for x in link_dict:
            link_dict[x].append(J[i])
            i += 1

        return link_dict


    def chi_random_init(self, link_dict):

        '''
        inserts random values for chi (MF-parameter) into the the link dictionary. 
        
        Parameters
        ---------
        link_dict: dictionary 
                -> dictionary conatining the links
                -> values must be appendable lists
        
        Returns
        ------
        link_dict: dictionary 
                same as input but with random chis appended at the end of each value
                -> chis are complex with real and imaginary part in [0, 1)
                -> chis are appended to the last postion of each value 
        '''
        
        
        chi = 1*self.rng.random(int(3*self.T*(self.T+1)/2)) + 1j*self.rng.random(int(3*self.T*(self.T+1)/2))
        # chi = 1j+self.rng.random(int(3*self.T*(self.T+1)/2)) # imaginary initialization
        # chi = 1*self.rng.random(int(3*self.T*(self.T+1)/2)) # real initialization
        # chi = np.ones(int(3*self.T*(self.T+1)/2))*(1-0.00001*1j)

        i = 0
        
        for x in link_dict:
            link_dict[x].append(chi[i])
            i += 1

        return link_dict

    def chi_pi_phase_init(self, link_dict, orient):
        
        for x in link_dict:
            link_dict[x].append(1/np.sqrt(6))

            
        if orient == 'down': 

            for c in np.arange(1, self.T + 1): # loop over every column except last one

                for i in np.arange(1, int(np.floor(c/2))+1): # loop over every second site in column 
                
                    s = int(c*(c-1)/2 + 2*i) # site number
                
                    link_dict[str(s) + str(int(s+c))][3] = -1/np.sqrt(6) # link s -> bottom right neighbour of s
                    link_dict[str(int(s+c)) + str(int(s+c+1))][3] = -1/np.sqrt(6) # link bottom right neighbour of s -> top right neighbour of s

        elif orient == 'up':
                    
            for c in np.arange(1, self.T +2): # loop over !every! column

                for i in np.arange(1, int(np.floor(c/2)+1)): # loop over every secons site in column

                    s = int(c*(c-1)/2 + 2*i) # site number

                    link_dict[str(int(s-c)) + str(s)][3] = -1/np.sqrt(6) # link bottom left neighbour of s -> s

                    if c != self.T +1:

                        link_dict[str(s) + str(s+c)][3] = - 1/np.sqrt(6) # link s -> bottom right neighbour of s (only exists if not i nlast column)



    def chi_VBS_init(self, link_dict):

        '''
        only works for T = 3!
        '''

        if self.T != 3:
            print("Error: VBS initialization only works for T = 3, input: T =", self.T, ".")
            quit()
        
        for x in link_dict:
            link_dict[x].append(0)

        link_dict['12'][3] = 1
        link_dict['36'][3] = 1
        link_dict['45'][3] = 1
        link_dict['78'][3] = 1
        link_dict['910'][3] = 1

        
        

    def mu_init(self):

        '''
        creates random vlaues for mu (Placeholder-ish)

        Returns
        ------
        mu_arr = array of random values in [0,1)
                -> length (T+1)(T+2)/2
        '''
        mu_arr = self.rng.random(int((self.T+1)*(self.T+2)/2)) #random positive initialization
        # mu_arr = np.zeros(int((self.T+1)*(self.T+2)/2)) # zero intialization
        # mu_arr = np.full(int((self.T+1)*(self.T+2)/2), 0.854)

        return mu_arr 


    def Ham_builder(self, pop_link_dict, mu_arr):

        '''
        builds the Hamiltonian

        Parameters
        ---------
        pop_link_dict: dictionary
                dictionary of all links in the system
                -> values are [a,b,c,d] with a->b the sites connected by the link and c,d chi and J (order arbitrary)
        mu_arr: array 
                array of mus (chemical potentials)
                length: (T+1)(T+2)/2

        Returns
        ------
        Ham: array 
                upper triangle of the Hamiltonian
        '''

        Ham = np.zeros((int((self.T+1)*(self.T+2)/2), int((self.T+1)*(self.T+2)/2)), dtype = np.complex128)
        np.fill_diagonal(Ham, mu_arr)

        for x in pop_link_dict: 
            a, b, J, chi = pop_link_dict[x] # link a -> b (<=> a<b)
            Ham[a-1, b-1]  = (-J*(1+self.kappa/2)+J*self.kappa*(np.absolute(chi))**2)*np.conjugate(chi)/2 # only fills upper triangle


        return Ham



        










    


#print(uNN_dict_gen(T))
    
   
    
    
    
    
    
    
    
    
    
    
    
