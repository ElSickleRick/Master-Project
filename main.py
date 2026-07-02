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
T  = 10# # triangles in base
kappa = 10 # biquadratic exchange constant
beta = 200 # inverse temperatur


'initialization parameters'
chi_init = "real" # variants for initializing chi, for options see below

# options 0/"complex": complex, 1/"real": real, "up": 0/pi-flux phase with pi flux in up-triangles, "down": 0/pi-flux phase with pi flux in down-triangles, "VBS": valence bond solid (only for T=3!)

'convergence parameters:'
mu_step_base = 0.3 # mean step size of the chemical potential 
mu_rm_scale = 0.1 # maximum size of fluctuations in both directions around the means step for mu (so interval is mu_step_base +- mu_rm_scale)
chi_rm_scale = 0.2 # maximum value of random mixing parameter for MF-bond-parameters
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
project_name = "up_T=40_0207_01"
data_miner.zero_T_iter(seed, rng, T, kappa, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras)
"""


project_name = "zero_0207_01"
target_con = True
target = 0
data_miner.cond_size_iter(seed, rng, beta, kappa, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras, target_con, target)

"""
init = system_init(T, kappa, rng)
link_dict, plaqu_dict, mu_arr, pop_link_dict = init.init_master(chi_init)

meth = methods(T, beta, kappa)
mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, free_en_hist, eival, eivec, sc_iter = meth.MF_solver(rng, iter_paras, pre_ana_paras, convergence_paras, link_dict, plaqu_dict, mu_arr, pop_link_dict)

post_ana = post_analysis(T ,kappa, beta, plaqu_dict, pop_link_dict, eival, eivec, mu_hist_dict, mu_arr, bond_hist_dict, plaqu_hist_dict, sc_iter)
post_ana.MF_iter_plot()
post_ana.real_space_plot() 
post_ana.mu_dist_plot()

plt.plot(free_en_hist)
"""
 

plt.show()











