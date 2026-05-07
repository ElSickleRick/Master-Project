import numpy as np
import time 

from system_init import system_init 
from MF_loop import MF_loop 


  

t = time.time()

T  = 200
kappa = 3

init = system_init(T, kappa)
MF = MF_loop() 

link_dict = init.link_dict_gen()

mu_arr = init.mu_init()

pop_link_dict = init.J_init(init.chi_init(link_dict))

Ham = init.Ham_builder(pop_link_dict, mu_arr)
    
print(Ham)

print("execution time", time.time()-t ," s for T =", T)

