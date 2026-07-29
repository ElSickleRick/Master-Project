import numpy as np
import time


class system_init:

    def __init__(self,T, kappa, rng, C, mag_elas, theta): 

        self.T = T
        self.kappa = kappa
        self.rng = rng
        self.C = C
        self.mag_elas = mag_elas
        self.theta = theta

        return
   
    """
    chi_init options include 
    0/"complex": random complex, 
    1/"real": random real, 
    "up": 0/pi-flux phase with pi flux in up-triangles, 
    "down": 0/pi-flux phase with pi flux in down-triangles, 
    "zero": zero flux in all plaquettes, 
    "pi": pi flux in all plaquettes, 
    "pi/2": pi/2 flux in every plaquette
    "-pi/2" : -pi/2 flux in every plaquette
    "VBS": valence bond solid (only for T=3!)
    """


    def init_master(self, chi_init):
        '''
        chi convention: the entry for link a -> b contains chi_{ab} ( ~ <c_a^\dag c_b>)
        '''
        
        link_dict = self.link_dict_gen()
        plaqu_dict = self.plaqu_dict_gen()
        strain_cord_dict = self.strain_cord_gen()
        pop_link_dict = self.strain_calc(strain_cord_dict, link_dict)

        if chi_init == "complex" or chi_init == 0:
            chi = 1*self.rng.random(int(3*self.T*(self.T+1)/2)) + 1j*self.rng.random(int(3*self.T*(self.T+1)/2)) # complex initialization

            i = 0
        
            for x in link_dict:
                link_dict[x].append(chi[i])
                i += 1

            mu_arr = self.mu_init(0)

        elif chi_init == "real" or chi_init == 1:
            chi = 1*self.rng.random(int(3*self.T*(self.T+1)/2)) # real initialization

            i = 0
        
            for x in link_dict:
                link_dict[x].append(chi[i])
                i += 1

            mu_arr = self.mu_init(0)

        elif chi_init == "up":
            self.chi_dqsl_init(pop_link_dict, 'up')
            mu_arr = self.mu_init(0)

        elif chi_init == "down":
            self.chi_dqsl_init(pop_link_dict, 'down')
            mu_arr = self.mu_init(0)

        elif chi_init == "zero":
            self.chi_zero_flux_init(pop_link_dict)
            mu_arr = self.mu_init(-0.67574)

        elif chi_init == "pi":
            self.chi_pi_flux_init(pop_link_dict)
            mu_arr = self.mu_init(0.67574)

        elif chi_init == "pi/2":
            self.chi_pi_half_flux_init(link_dict)
            mu_arr = self.mu_init(0)

        elif chi_init == "-pi/2":
            self.chi_minus_pi_half_flux_init(link_dict)
            mu_arr = self.mu_init(0)

        elif chi_init == "VBS":
            self.chi_VBS_init(pop_link_dict)
            mu_arr = self.mu_init(0)

        else:

            #pop_link_dict['12'].append(1/2*np.exp(4j*np.pi/7))
            #pop_link_dict['23'].append(1/2*np.exp(4j*np.pi/7))
            #pop_link_dict['13'].append(1/2*np.exp(8j*np.pi/7))

            for x in link_dict:
                link_dict[x].append(1/np.sqrt(6)*np.exp(1j*0.0000000001))


            for c in np.arange(1, self.T +2): # loop over !every! column

                for i in np.arange(1, int(np.floor(c/2)+1)): # loop over every secons site in column

                    s = int(c*(c-1)/2 + 2*i) # site number

                    link_dict[str(int(s-c)) + str(s)][3] = -1/(2*np.sqrt(6)*np.exp(1j*0.0000000001)) # link bottom left neighbour of s -> s

                    if c != self.T +1:

                        link_dict[str(s) + str(s+c)][3] = - 1/(2*np.sqrt(6)*np.exp(1j*0.0000000001)) # link s -> bottom right neighbour of s (only exists if not i nlast column)

        return link_dict, strain_cord_dict, plaqu_dict, mu_arr, pop_link_dict


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

        """
        generates a dictionary containing information on all plaquettes of the system

        Returns
        -------
        plaqu_dict: dictionary
                    the entries are two-component arrays, the first entrie being the orientation of the triangle (either "up" or "down"), the second ent        ry is an array of length 3 containing the indices of all points of the plaquette in ascending order  

        """

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


    def strain_cord_gen(self):
        
        s = int((self.T+1)*(self.T+2)/2) # # sites
        c_max = self.T # maximum column number (column counting starts at 0 here!!)
 
        str_cord_dict = {}

        for i in range(1, s+1):

            c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))-1) # caclulate column number c
            p = int(i-c*(c+1)/2-1) # calculate position p inside column c 
            
            # calculate the unstrained position of atom i:
            x = c - p/2 - self.T/2
            y = np.sqrt(3)*p/2 - self.T/(2*np.sqrt(3))

            # calcualte strained postions of atom i (the strain strength gets scaled with the linear system size):
            x_str = x + self.C*((x**2-y**2)*np.sin(2*self.theta) + 2*x*y*np.cos(2*self.theta))/self.T
            y_str = y + self.C*((x**2-y**2)*np.cos(2*self.theta) - 2*x*y*np.sin(2*self.theta))/self.T

            str_cord_dict.update({i : [[x,y], [x_str, y_str]]})

        return(str_cord_dict)


    def strain_calc(self, str_cord_dict, link_dict):

        for i in link_dict:

            s, e = link_dict[i]

            bond_len = np.linalg.norm( np.array(str_cord_dict[s][1]) - np.array(str_cord_dict[e][1]) ) # length of strained bond
            # J = (1-self.mag_elas*(bond_len - 1))
            J = np.exp(-self.mag_elas*(bond_len -1))


            if J > 0:

                link_dict[i].append(J) 
            else:

                print("error; strain produces ferromagnetic couplings; programm stopped")
                quit() 

        return link_dict
            

    def mu_init(self, mu):

        '''
        creates random vlaues for mu (Placeholder-ish)

        Returns
        ------
        mu_arr = array of random values in [0,1)
                -> length (T+1)(T+2)/2
        '''
        # mu_arr = self.rng.random(int((self.T+1)*(self.T+2)/2)) #random positive initialization
        # mu_arr = np.zeros(int((self.T+1)*(self.T+2)/2)) # zero intialization
        mu_arr = np.full(int((self.T+1)*(self.T+2)/2), mu)

        return mu_arr



    def chi_dqsl_init(self, link_dict, orient):
        
        for x in link_dict:
            link_dict[x].append(0.2)

            
        if orient == 'down': 

            for c in np.arange(1, self.T + 1): # loop over every column except last one

                for i in np.arange(1, int(np.floor(c/2))+1): # loop over every second site in column 
                
                    s = int(c*(c-1)/2 + 2*i) # site number
                
                    link_dict[str(s) + str(int(s+c))][3] = -0.2 # link s -> bottom right neighbour of s
                    link_dict[str(int(s+c)) + str(int(s+c+1))][3] = -0.2 # link bottom right neighbour of s -> top right neighbour of s

        elif orient == 'up':
                    
            for c in np.arange(1, self.T +2): # loop over !every! column

                for i in np.arange(1, int(np.floor(c/2)+1)): # loop over every secons site in column

                    s = int(c*(c-1)/2 + 2*i) # site number

                    link_dict[str(int(s-c)) + str(s)][3] = -0.2 # link bottom left neighbour of s -> s

                    if c != self.T +1:

                        link_dict[str(s) + str(s+c)][3] = -0.2 # link s -> bottom right neighbour of s (only exists if not i nlast column)

    def chi_pi_half_flux_init(self, link_dict):
        
        chi = 0.200169

        s = int((self.T+1)*(self.T+2)/2) # number of sites
        c_max = self.T + 1 # maximum column number (see below)
                
        for i in range(1, s+1): # loop over every site
            
            c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))) # caclulate column number c of site i
            s = int(i - (c-1)*c/2) # position of site i in column c 

            if s % 2 == 1: #type a sites (every other site)

                if c != c_max: 
                    link_dict[str(i) + str(int(i+c+1))].append(-chi) # neighbour above + right 
                    link_dict[str(i) + str(int(i+c))].append(-1j*chi) # neighbour right 

                if i != int(c*(c+1)/2):
                    link_dict[str(i) + str(i+1)].append(1*chi) # neighbour above

            elif s % 1 == 0: # type b sites (betweenbetween  every other site)

                if c!= c_max:
                    link_dict[str(i) + str(int(i+c+1))].append(chi) # neighbour above + right 
                    link_dict[str(i) + str(int(i+c))].append(1j*chi) # neighbur right 

                if i != int(c*(c+1)/2):
                    link_dict[str(i) + str(i+1)].append(chi) # neighbour above

    def chi_minus_pi_half_flux_init(self, link_dict):
        
        chi = 0.200169

        s = int((self.T+1)*(self.T+2)/2) # number of sites
        c_max = self.T + 1 # maximum column number (see below)
                
        for i in range(1, s+1): # loop over every site
            
            c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))) # caclulate column number c of site i
            s = int(i - (c-1)*c/2) # position of site i in column c 

            if s % 2 == 1: #type a sites (every other site)

                if c != c_max: 
                    link_dict[str(i) + str(int(i+c+1))].append(chi) # neighbour above + right 
                    link_dict[str(i) + str(int(i+c))].append(-1j*chi) # neighbour right 

                if i != int(c*(c+1)/2):
                    link_dict[str(i) + str(i+1)].append(1*chi) # neighbour above

            elif s % 1 == 0: # type b sites (betweenbetween  every other site)

                if c!= c_max:
                    link_dict[str(i) + str(int(i+c+1))].append(chi) # neighbour above + right 
                    link_dict[str(i) + str(int(i+c))].append(1j*chi) # neighbur right 

                if i != int(c*(c+1)/2):
                    link_dict[str(i) + str(i+1)].append(-chi) # neighbour above
   
    def chi_zero_flux_init(self, link_dict):

        for x in link_dict:
            link_dict[x].append(0.164712)

    def chi_pi_flux_init(self, link_dict):

        for x in link_dict:
            link_dict[x].append(-0.164712)

    def chi_VBS_init(self, link_dict):

        '''
        only works for T = 3!
        '''

        if self.T != 3:
            print("Error: VBS initialization only works for T = 3, input: T =", self.T, ".")
            quit()
        
        for x in link_dict:
            link_dict[x].append(0)

        link_dict['12'][3] = 0.5
        link_dict['36'][3] = 0.5
        link_dict['45'][3] = 0.5
        link_dict['78'][3]= 0.5
        link_dict['910'][3] = 0.5


    def noise_machine(self, pop_link_dict, mu_arr, chi_noise_scale, mu_noise_scale):
        
        chi_bound = 0.05
        mu_bound = 0.05

        for x in pop_link_dict:
            s, e, J, chi = pop_link_dict[x]
            chi_real = np.real(chi)
            chi_imag = np.imag(chi)
            pop_link_dict[x][3] += chi_noise_scale*(self.rng.uniform(-1,1)*max(np.absolute(chi_real), chi_bound) + 1j*self.rng.uniform(-1,1)*max(np.absolute(chi_imag), chi_bound))

        for i in range(0, len(mu_arr)-1):
            mu_arr[i] += mu_noise_scale*self.rng.uniform(-1,1)*max(np.absolute(mu_arr[i]), mu_bound)
        
        return mu_arr

        
   
    
    
    
    
    
    
    
    
    
    
    
