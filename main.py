import numpy as np
import time 

from system_init import system_init 
from MF_loop import MF_loop 


  

t = time.time()

T  = 3
kappa = 3
beta = 100
N_diff_bound = 0.001
mu_step = 0.5

init = system_init(T, kappa)
MF = MF_loop() 

link_dict = init.link_dict_gen()

mu_arr = init.mu_init()

pop_link_dict = init.J_init(init.chi_init(link_dict))

Ham = init.Ham_builder(pop_link_dict, mu_arr)
    
print("time for building the Hamiltonian:", time.time()-t ," s for T =", T)

eival, eivec = MF_loop.Ham_diag(Ham)

weights, Z = MF_loop.thermal_calc(eival, beta)



print(eival)
print("Z is", Z)
print(mu_arr)
print(mu_update)

