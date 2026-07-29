import numpy as np
# import numba as nb
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import os
import pickle

from system_init import system_init 
from methods import methods
from analysis import pre_analysis, post_analysis
from data_acquisition import save_data, data_miner

seed = np.random.SeedSequence().entropy # pull rng initialization seed from system entropy
# seed = 13082203210817060306 # fixed rng initializtaion seed 
rng = np.random.default_rng(seed)

# look up: T | # sites:   13|105  21|253  30|496  37|741  43|990  62|2016

'Model parameters:'
T  = 20 # # triangles in base
kappa = 10 # biquadratic exchange constant
beta = 200 # inverse temperatur
C = 0 # 0.5 # strain strength x linear sytem size 
mag_elas = 0 # 2 # magneto-elastic coupling
theta = 0 # rotation angle of the strain pattern (np.pi/2)

'initialization parameters'
chi_init = "complex" # variants for initializing chi, for options see system_init
chi_noise_scale = 0 #0.025 # scale of the noise applied to the MF parameters relative to their real/imaginary part
mu_noise_scale = 0 #0.025 # sclae of noise applied to chemical potentinals relative to their absolute value

'convergence parameters:'
mu_step_base = 1.1 # 1.1 #0.5   # mean step size of the chemical potential 
mu_rm_scale = 0.35  # maximum size of fluctuations in both directions around the means step for mu (so interval is mu_step_base +- mu_rm_scale)
chi_rm_scale = 0.3 # maximum value of random mixing parameter for MF-bond-parameters
N_dif_bd = 0.0001 # maximum tolerance for deviation of local particle number from 1 (usually 0.0001)
chi_dif_bd = 0.0001 # bound for convergence of absolute value of Chi (MF-bond-parameter) (usually 0.0001)
max_iter_cond = True # if True, self consistency loop will terminate prematurely after a certain number of steps
sc_iter_max = 5000 # maximum number of iterations before the self-consistency loop will terminate prematurely (requires max_iter_cond = True)

iter_paras = [max_iter_cond, sc_iter_max]
convergence_paras = [mu_step_base, mu_rm_scale, chi_rm_scale, N_dif_bd, chi_dif_bd]

'analysis parameters:'
mu_length = 10 # number of chemical potentials that are plotted
bond_length = 10 # number of MF-bond-parameters that are plotted
plaqu_length = 10 # number of plaquettes that are plotted

pre_ana_paras = [mu_length, bond_length, plaqu_length]

"""

project_name = "pi_T=30_0607_01"
data_miner.zero_T_iter(seed, rng, T, kappa, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras)

""" """  

chi_init = "-pi/2"
project_name = "minus_pi_half_1407_01"
target_con = True
target = -np.pi/2
data_miner.cond_size_iter(seed, rng, beta, kappa, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras, target_con, target)

""" 

init = system_init(T, kappa, rng, C, mag_elas, theta)
link_dict, strain_cord_dict, plaqu_dict, mu_arr, pop_link_dict = init.init_master(chi_init)
mu_arr = init.noise_machine(pop_link_dict, mu_arr, chi_noise_scale, mu_noise_scale)

meth = methods(T, beta, kappa, C, mag_elas)
mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, free_energy_hist, eival, eivec, sc_iter = meth.MF_solver(rng, iter_paras, pre_ana_paras, convergence_paras, link_dict, plaqu_dict, mu_arr, pop_link_dict)

post_ana = post_analysis(T ,kappa, beta, C, mag_elas, strain_cord_dict, plaqu_dict, pop_link_dict, eival, eivec,  mu_arr)
post_ana.MF_iter_plot(sc_iter, mu_hist_dict, bond_hist_dict, plaqu_hist_dict)
post_ana.real_space_plot() 
post_ana.flux_hist_plot()
post_ana.free_energy_iter_plot(sc_iter, free_energy_hist)

""" 

project_name  = "T=45_rot_2707_01"
target_con = True
target = np.pi/2
data_miner.cond_strain_iter(seed, rng, T, kappa, beta, mag_elas, theta, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras, target_con, target)

"""

plt.show()






