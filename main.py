import numpy as np
# import numba as nb
import time 
import scipy.linalg as la 
import matplotlib.pyplot as plt 

from system_init import system_init 
from MF_loop import MF_loop
from analysis import analysis


# look up: T | # sites:   13|105  21|253  30|496  37|741  43|990  62|2016

'Model parameters:'
T  = 6 # # triangles in base
kappa = 10 # biquadratic exchange constant
beta = 100 # inverse temperatur

'self consistency loop parameters:'
N_dif_bd = 0.0001 # maximum tolerance for deviation of local particle number from 1 => 0.001?
mu_step = 0.3 # maximum bond for random mixing parameter for the chemical potentials 
Chi_dif_bd = 0.0001 # bound for convergence of absolute value of Chi (MF-parameter)
rm_scale = 0.1 # maximum bound for random mixing parameter
max_iter_cond = True # if True, self consistency loop will terminate prematurely after a certain number of steps
sc_iter_max = 5000 # maximum number of iterations before the self-consistency loop will terminat prematurely (requires max_iter_cond == True)

'analysis parameters:'
mu_length = 7 # number of chemical potentials that are plotted
if mu_length > int((T+1)*(T+2)/2): mu_length = int((T+1)*(T+2)/2)
bond_length = 7 # number of bond MF-parameters that are plotted
if bond_length > int(3*T*(T+1)/2): bond_length = int(3*(T+1)*T/2)

init = system_init(T, kappa)

link_dict = init.link_dict_gen()
plaqu_dict = init.plaqu_dict_gen()


mu_select = np.random.choice(np.arange(0, int((T+1)*(T+2)/2)), mu_length, replace = False) # select mu_s to plot
mu_hist_dict = {}
for i in mu_select:
    mu_hist_dict.update({i: []})
bond_select = np.random.choice(list(link_dict.keys()), bond_length, replace = False) # select bonds to plot
bond_hist_dict = {}
for i in bond_select:
    bond_hist_dict.update({i: [link_dict[i][0], link_dict[i][1]]})



mu_arr = init.mu_init()
pop_link_dict = init.J_init(link_dict) # link dict containing the values for J and Chi on each bond
pop_link_dict = init.chi_init(pop_link_dict)

# init.chi_pi_phase_init(pop_link_dict, 'down')


print(pop_link_dict)


conv = False 
interrupt = False 
sc_iter = 0 # number of iterations of self-consistency loop 

while conv == False:
    
    conv = True
   
    Ham = init.Ham_builder(pop_link_dict, mu_arr)
    eival, eivec = la.eigh(Ham, lower = False) # daigonalize Hamiltonian, entries are in upper traingle! 

    MF = MF_loop(T, beta, eival, eivec, N_dif_bd, mu_hist_dict, mu_step, pop_link_dict, bond_hist_dict, Chi_dif_bd, rm_scale, conv)

    fe_di = np.array([MF.lin_fe_di(beta*x) for x in eival]) # calculate fermi_dirac distributions

    mu_arr, conv  = MF.update(mu_arr, fe_di) # fermi_dirac distributions

    sc_iter += 1

    if sc_iter >= sc_iter_max and max_iter_cond == True:
        print("self consistency loop intrrupted: to many iterations (",sc_iter_max,")")
        interrupt = True 
        conv = True 
        break

    if (sc_iter-1) % 250 == 0 and sc_iter != 1:
        print("finished iteration", sc_iter-1)
        

if interrupt == False:
    print("self-consistency loop finished normally after", sc_iter, "iterations")


ana = analysis(T, kappa, beta, plaqu_dict, pop_link_dict, eival, eivec, mu_hist_dict,  mu_arr, bond_hist_dict, sc_iter)

F = ana.free_en_calc()

print("free energy density:", F)


ana.real_space_plot()
ana.MF_iter_plot()
#ana.DOS_hist()

#print(pop_link_dict)
#print(Ham)
#print("mu array:", mu_arr)
#print("eival:", eival)


plt.show()



