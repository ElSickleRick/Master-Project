import numpy as np
# import numba as nb
import time 
import scipy.linalg as la 
import matplotlib.pyplot as plt 

from system_init import system_init 
from MF_loop import MF_loop
from analysis import analysis


T  = 7 # # triangles in base
kappa = 10 # biquadratic exchange constant
beta = 10 # inverse temperatur

N_diff_bd = 0.01 # maximum tolerance for deviation of local particle number from 1 => 0.001?
mu_step = 0.75 # stepsize for chemical potential in [0,1)  ??? I am not ssure why I put this in ???
Chi_dif_bd = 0.01 # bound for convergence of absolute value of Chi (MF-parameter)
sc_iter_max = 150 # maximum number of iterations before the self-consistency loop will self-terminate


init = system_init(T, kappa)

link_dict = init.link_dict_gen()

mu_arr = init.mu_init()

pop_link_dict = init.chi_init(init.J_init(link_dict))

conv = False 
interrupt = False 
sc_iter = 0 # number of iterations of self-consistency loop 

while conv == False:
    
    conv = True
   
    Ham = init.Ham_builder(pop_link_dict, mu_arr)
    eival, eivec = la.eigh(Ham, lower = False) # daigonalize Hamiltonian 

    MF = MF_loop(beta, eival, eivec, N_diff_bd, mu_step, pop_link_dict, Chi_dif_bd)

    weights, Z = MF.thermal_calc() # calculate Boltzman-weights and partition fucntion

    # print("weights are:", weights) 
    
    temp, mu_arr = MF.mu_update(mu_arr, Z, weights) # update chemical potentials

    if temp == False:
        conv = False 

    for x in pop_link_dict: # update MF-parameters (parallelizable!) 
        temp = MF.Chi_update(x, weights, Z)

        if x == "58":
            print(" Chi on link 58 is", np.absolute(pop_link_dict["58"][3]), "(absolute value)", np.angle(pop_link_dict["58"][3])/np.pi, "(phase)")    
        if x == "89":
            print(" Chi on link 89 is", np.absolute(pop_link_dict["89"][3]), "(absolute value)", np.angle(pop_link_dict["89"][3])/np.pi, "(phase)")  
        if x == "59":
            print(" Chi on link 59 is", np.absolute(pop_link_dict["59"][3]), "(absolute value)", np.angle(pop_link_dict["59"][3])/np.pi, "(phase)") 


        if temp == False:
            conv = False 

    sc_iter += 1
    if sc_iter >= sc_iter_max:
        print("self consistency loop intrrupted: to many iterations")
        interrupt = True 
        conv = True 

    print("iteration", sc_iter, "finished, verdict:", conv)
    # print("N array is" , N_arr)
    # print(pop_link_dict)

if interrupt == False:
    print("self-consistency loop finished normally")
    print("number of iterations:", sc_iter)

    ana = analysis(pop_link_dict)

    mean, std, Chi_abs_arr = ana.Chi_abs_dist_plot()

    print("mean is", mean, "with standard deviation", std)

    plt.show()
