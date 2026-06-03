import numpy as np
# import numba as nb
import scipy.linalg as la 
import matplotlib.pyplot as plt 

from system_init import system_init 
from MF_loop import MF_loop
from analysis import pre_analysis, analysis

seed = np.random.SeedSequence().entropy # pull rng initialization seed from system entropy
# seed = 13082203210817060306 # fixed rng initializtaion seed 
rng = np.random.default_rng(seed)

# look up: T | # sites:   13|105  21|253  30|496  37|741  43|990  62|2016

'Model parameters:'
T  = 5 # # triangles in base
kappa = 10 # biquadratic exchange constant
beta = 100 # inverse temperatur


'initialization parameters'
chi_init = "A" # variants for initializing chi, for options see below

# options 0/"complex": complex, 1/"real": real, "up": 0/pi-flux phase with pi flux in up-triangles, "down": 0/pi-flux phase with pi flux in down-triangles, "VBS": valence bond solid (only for T=3!)

'self consistency loop parameters:'
N_dif_bd = 0.0001 # maximum tolerance for deviation of local particle number from 1 (usually 0.0001)
mu_step = 0.5 # maximum bond for random mixing parameter for the chemical potentials 
Chi_dif_bd = 0.0001 # bound for convergence of absolute value of Chi (MF-bond-parameter) (usually 0.0001)
rm_scale = 0.3 # maximum bound for random mixing parameter
max_iter_cond = True # if True, self consistency loop will terminate prematurely after a certain number of steps
sc_iter_max = 50000 # maximum number of iterations before the self-consistency loop will terminate prematurely (requires max_iter_cond = True)

'analysis parameters:'
mu_length = 7 # number of chemical potentials that are plotted
bond_length = 7 # number of MF-bond-parameters that are plotted
plaqu_length = 7 # number of plaquettes that are plotted


free_en_hist = []

def MF_solver():

    init = system_init(T, kappa, rng, chi_init)
    link_dict, plaqu_dict, mu_arr, pop_link_dict = init.init_master()

    pre_ana = pre_analysis(T, mu_length, link_dict, bond_length, plaqu_dict, plaqu_length)
    mu_hist_dict, bond_hist_dict, plaqu_hist_dict = pre_ana.hist_dict_gen()

    conv = False 
    interrupt = False 
    sc_iter = 0 # number of iterations of self-consistency loop 

    while conv == False:
    
        conv = True
   
        Ham = init.Ham_builder(pop_link_dict, mu_arr)
        eival, eivec = la.eigh(Ham, lower = False) # daigonalize Hamiltonian, entries are in upper traingle! 

        MF = MF_loop(T, beta, eival, eivec, N_dif_bd, mu_hist_dict, mu_step, pop_link_dict, bond_hist_dict, Chi_dif_bd, rm_scale, plaqu_hist_dict, conv, rng)
        fe_di = np.array([MF.lin_fe_di(beta*x) for x in eival]) # calculate fermi_dirac distributions

        mu_arr, conv  = MF.update(mu_arr, fe_di) # fermi_dirac distributions

        sc_iter += 1
        
        ana = analysis(T, kappa, beta, fe_di, plaqu_dict, pop_link_dict, eival, eivec, mu_hist_dict,  mu_arr, bond_hist_dict, plaqu_hist_dict, sc_iter)
        f = ana.free_en_calc()
        free_en_hist.append([f])

        if sc_iter >= sc_iter_max and max_iter_cond == True:
            print("self consistency loop intrrupted: to many iterations (",sc_iter_max,")")
            interrupt = True 
            conv = True 
            break

        if (sc_iter-1) % 500 == 0 and sc_iter != 1:
            print("finished iteration", sc_iter-1)
        

    if interrupt == False:
        print("self-consistency loop finished normally after", sc_iter, "iterations")

    
    return(link_dict, plaqu_dict, mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, eival, eivec, fe_di, sc_iter)


link_dict, plaqu_dict, mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, eival, eivec, fe_di, sc_iter= MF_solver()
ana = analysis(T, kappa, beta, fe_di, plaqu_dict, pop_link_dict, eival, eivec, mu_hist_dict,  mu_arr, bond_hist_dict, plaqu_hist_dict, sc_iter)


plt.scatter(np.arange(0,sc_iter), free_en_hist)

f = ana.free_en_calc()
print("free energy density", f)

ana.real_space_plot()
ana.MF_iter_plot()
# ana.Chi_ph_dist_plot()
# ana.Chi_abs_dist_plot()
# ana.DOS_hist()
energy = ana.en_calc()
#path = ana.Chi_path_plot()

#print(pop_link_dict)
#print(Ham)
#print(fe_di)
#print("mu array:", mu_arr)
#print("eival:", eival)
#print(np.absolute(eivec))

print("energy:", energy)

# for x in pop_link_dict:
    # print(x, "absolute value", np.absolute(pop_link_dict[x][3]), "phase", np.angle(pop_link_dict[x][3]))



plt.show()



