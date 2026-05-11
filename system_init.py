import numpy as np
import time


class system_init:

    def __init__(self,T, kappa): 

        '''
        Parameters
        ----------
        T: int
            # triangles in base = # sites in base -1
            ->minimum is T = 1 
        kappa: float   
            strength of the biquadratic exchange constant

        '''

        self.T = T
        self.kappa = kappa


    def NN_dict_gen(self): 
    
        '''
        Parameters
        ----------

        Returns
        -------
        NN_dict : dictionarie of NN of all sites 
        -> keys are site numbers
        -> values are lists of NN sites
        -> counting starts at 1 and goes to (T+1)*(T+2)/2
        '''
        
        s = int((self.T+1)*(self.T+2)/2) # number of sites
        c_max = self.T + 1 # maximum column number (see below)
        
        NN_dict = {1 : [2,3]}
        
        for i in range(2, s+1): 
            
            NN = [] # save NN for site i here
            
            c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))) # caclulate column number c of site i
            p = int(i - (c-1)*c/2) # calcualte position p of site i inside coloumn 
            
            if p != 1: # neighbor below and below + left 
                NN.extend([int(i-1), int(i-c)])
            
            if i != c*(c+1)/2: # neighbor above and above + left 
                NN.extend([int(i+1), int(i-c+1)])
                
            if c != c_max: # neighbor above + right and below + right 
                NN.extend([int(i+c), int(i+c+1)])
            
            NN_dict.update({i : NN})
        
        return NN_dict

    def uNN_dict_gen(self):

        '''
        Only allows unique nearest neighbors (uNN), such that there is no redundancy.
        -> uNN are always above, above + right and below + right sites if existent

        Parameters
        ----------

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

        Parameters
        ----------

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

    def J_init(self, link_dict):

        '''
        Placeholder: inserts random J's from [0,1) into the link dictionary 

        '''

        J = 2*np.random.rand(int(3*self.T*(self.T+1)/2))
        i = 0
        
        for x in link_dict:
            link_dict[x].append(J[i])
            i += 1

        return link_dict

    def chi_init(self, link_dict):

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
        
        chi = 10*np.random.rand(int(3*self.T*(self.T+1)/2)) + 1j*np.random.rand(int(3*self.T*(self.T+1)/2))
        i = 0
        
        for x in link_dict:
            link_dict[x].append(chi[i])
            i += 1

        return link_dict

    def mu_init(self):

        '''
        creates random vlaues for mu (Placeholder-ish)

        Parameters
        ----------

        Returns
        ------
        mu_arr = array of random values in [0,1)
                -> length (T+1)(T+2)/2
        '''
        # mu_arr = np.random.rand(int((self.T+1)*(self.T+2)/2)) #random initialization
        mu_arr = np.zeros(int((self.T+1)*(self.T+2)/2))

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
            a, b, c, d = pop_link_dict[x] # link a -> b (<=> a<b)

            if c.dtype == np.complex128: # figure out wich value is J and which is chi. Maybe this can be simplified later 
                J = c
                chi =d
            else:
                J = d   
                chi = c


            Ham[a-1, b-1] = temp = (-J*(1+self.kappa/2)+J*self.kappa*np.absolute(chi))*np.conjugate(chi)/2 # only fills uper triangle

        return Ham



        










    


#print(uNN_dict_gen(T))
    
   
    
    
    
    
    
    
    
    
    
    
    
