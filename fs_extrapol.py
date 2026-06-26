import numpy as np
# import numba as nb
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import os
import pickle

from system_init import system_init 
from MF_loop import MF_loop
from analysis import pre_analysis, analysis

# seed = np.random.SeedSequence().entropy # pull rng initialization seed from system entropy
seed = 13082203210817060306 # fixed rng initializtaion seed 
rng = np.random.default_rng(seed)

# look up: T | # sites:   13|105  21|253  30|496  37|741  43|990  62|2016

'Model parameters:'
#T  = 37 # # triangles in base
kappa = 10 # biquadratic exchange constant
beta = 150 # inverse temperatur


'initialization parameters'
chi_init = "down" # variants for initializing chi, for options see below

# options 0/"complex": complex, 1/"real": real, "up": 0/pi-flux phase with pi flux in up-triangles, "down": 0/pi-flux phase with pi flux in down-triangles, "VBS": valence bond solid (only for T=3!)

'self consistency loop parameters:'
mu_step_base = 0.4 # mean step size of the chemical potential 
mu_rm_scale = 0.1 # maximum size of fluctuations in both directions around the means step for mu (so interval is mu_step_base +- mu_rm_scale)
chi_rm_scale = 0.3 # maximum value of random mixing parameter for MF-bond-parameters
N_dif_bd = 0.0001 # maximum tolerance for deviation of local particle number from 1 (usually 0.0001)
Chi_dif_bd = 0.0001 # bound for convergence of absolute value of Chi (MF-bond-parameter) (usually 0.0001)
max_iter_cond = True # if True, self consistency loop will terminate prematurely after a certain number of steps
sc_iter_max = 5000 # maximum number of iterations before the self-consistency loop will terminate prematurely (requires max_iter_cond = True)

'analysis parameters:'
mu_length = 10 # number of chemical potentials that are plotted
bond_length = 10 # number of MF-bond-parameters that are plotted
plaqu_length = 10 # number of plaquettes that are plotted

path_head = "/home/kuerschner/Documents/Master-Project/data"

"""
projects = {
        "fs_extrapol_down_2606_01" : "down",
        "fs_extrapol_up_2606_01": "up" 
        }
"""

projects = {
        "fs_extrapol_down_2606_01": "down"
        }

fig, ax = plt.subplots(1,3)

for project in projects:

    project_path = os.path.join(path_head, project)
    
    inv_N = []
    mu_mean = []
    mu_std = []
    chi_abs_mean = []
    chi_abs_std = []

    for size in os.listdir(project_path):
        
        size_path = os.path.join(project_path, size)
        
        with open(os.path.join(size_path, "N.pkl"), "rb") as f:
            T = pickle.load(f)

        with open(os.path.join(size_path, "mu_arr.pkl"), "rb") as f:
            mu_arr = pickle.load(f)

        with open(os.path.join(size_path, "pop_link_dict.pkl"), "rb") as f:
            pop_link_dict = pickle.load(f)

        inv_N.append(2/((T+1)*(T+2)))

        mu_mean.append(np.mean(mu_arr))
        mu_std.append(np.std(mu_arr))

        chi_abs_arr = []

        for link in pop_link_dict:
            chi_abs_arr.append(np.absolute(pop_link_dict[link][3]))

        chi_abs_mean.append(np.mean(chi_abs_arr))
        chi_abs_std.append(np.std(chi_abs_arr))

    print(chi_abs_mean)
    print(chi_abs_std)

    ax[1].errorbar(inv_N, mu_mean, yerr = mu_std)
    #ax[0][0].title("mu")

    ax[2].errorbar(inv_N, chi_abs_mean, yerr = chi_abs_std)
    #ax[0][1].title("chi")

plt.show()
print(ax.shape)


