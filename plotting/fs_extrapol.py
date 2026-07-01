import numpy as np
# import numba as nb
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import os
import pickle

def lin_fe_di(Ebeta):

    '''
    linearization of fermi dirac distribution for the case that every energy is by default doubly degenerate (as it is in this case)
    '''

    if Ebeta < -2:
        return 2
         
    elif Ebeta > 2:
        return 0

    else:
        return (1/2)*(2-Ebeta)


def free_en_calc(T, beta, kappa, eival, mu_arr, pop_link_dict):
    '''
    calculates the free energy desnsity of a given mean field solution

    Returns:
    --------
    free_en: float 
            free energy  
    '''

    F = -(2/beta)*np.sum(np.log(1+np.exp(-beta*eival)))

    for x in pop_link_dict:
        s, e, J, Chi = pop_link_dict[x]
        F += 2*(J*(1+kappa/2)- 6*J*kappa*(np.absolute(Chi)**2))*(np.absolute(Chi)**2) 

    mu_part = -np.sum(mu_arr)

    return 2*(F + mu_part)/((T+1)*(T+2))


def energy_calc(T, beta, kappa, eival, fe_di, mu_arr, pop_link_dict):

    const = 0
    
    for x in pop_link_dict:
        s, e, J, Chi = pop_link_dict[x] 
        const += 2*((1+kappa/2)-6*kappa*np.absolute(Chi)**2)*np.absolute(Chi)**2


    return 2*(np.sum(eival*fe_di) - np.sum(mu_arr) + const)/((T+1)*(T+2)) 

    
def fit_fun(x, a):

    y = 0
    m = len(a)
        
    for i in range(0, m):

        y += a[i]*(x**(m-i-1))

    return y 


path_head = "/home/kuerschner/Documents/Master-Project/data"


projects = {
        #"fs_extrapol_down_2906_03" : "down",
        #"fs_extrapol_up_2906_03": "up",
        "fs_extrapol_down_0107_02": "down",
        "fs_extrapol_up_0107_02": "up",
        "fs_extrapol_pi_half_0107_01" : f"$\pi$/2"
            }

x_max = 0.003
fig, ax = plt.subplots(1,3)
fit_range = np.linspace(0, x_max, 1000)

for project in projects:

    project_path = os.path.join(path_head, project)
    
    inv_N = []
    mu_mean = []
    mu_stdm = []
    chi_abs_mean = []
    chi_abs_stdm = []
    free_energy_arr = []

    for size in os.listdir(project_path):
        
        size_path = os.path.join(project_path, size)
        
        with open(os.path.join(size_path, "mu_arr.pkl"), "rb") as f:
            mu_arr = pickle.load(f)

        with open(os.path.join(size_path, "pop_link_dict.pkl"), "rb") as f:
            pop_link_dict = pickle.load(f)

        with open(os.path.join(size_path, "eival.pkl"), "rb") as f:
            eival = pickle.load(f)

        with open(os.path.join(size_path, "eivec.pkl"), "rb") as f:
            eivec = pickle.load(f)

        with open(os.path.join(size_path, "info.pkl"), "rb") as f:
            info = pickle.load(f)

        T = info["T"]
        kappa = info["kappa"]
        beta = info["beta"]
        
        N = ((T+1)*(T+2))/2
        inv_N.append(1/N)

        mu_mean.append(np.mean(mu_arr))
        mu_stdm.append(np.std(mu_arr, ddof = 1)/(np.sqrt(N)))

        chi_abs_arr = []

        for link in pop_link_dict:
            chi_abs_arr.append(np.absolute(pop_link_dict[link][3]))

        chi_abs_mean.append(np.mean(chi_abs_arr))
        chi_abs_stdm.append(np.std(chi_abs_arr)/(np.sqrt(N)))

        fe_di = np.array([lin_fe_di(beta*x) for x in eival])

        free_energy_arr.append(free_en_calc(T, beta, kappa, eival, mu_arr, pop_link_dict))

    mu_fit = np.polyfit(inv_N, mu_mean, 1, w = [1/s for s in mu_stdm])

    ax[0].errorbar(inv_N, mu_mean, yerr = mu_stdm, fmt='x', label = str(projects[project]))
    ax[0].plot(fit_range, [fit_fun(x,mu_fit) for x in fit_range], label = f"fit {projects[project]}")
    ax[0].hlines(0, 0, x_max, linestyle = "--", color = 'k')


    chi_abs_fit = np.polyfit(inv_N, chi_abs_mean, 1, w = [1/s for s in chi_abs_stdm])

    ax[1].errorbar(inv_N, chi_abs_mean, yerr = chi_abs_stdm, fmt='x', label = str(projects[project]))
    ax[1].plot(fit_range, [fit_fun(x,chi_abs_fit) for x in fit_range], label = f"fit {projects[project]}")
    ax[1].hlines(0.19754, 0, x_max, linestyle = "--", color = 'k')
    
    free_energy_fit = np.polyfit(inv_N, free_energy_arr, 1)

    ax[2].scatter(inv_N, free_energy_arr, label = str(projects[project]))
    ax[2].plot(fit_range, [fit_fun(x,free_energy_fit) for x in fit_range], label = f"fit {projects[project]}")
    ax[2].hlines(-1.2206, 0, x_max, linestyle = "--", color = 'k')


    ax[0].set_ylabel("$\mu$", fontsize = "x-large")
    ax[1].set_ylabel("$|\chi|$", fontsize = "x-large")
    ax[2].set_ylabel("free energy", fontsize = "x-large")
    

for axes in ax:
    axes.set_xlim([0, x_max])
    axes.legend()

plt.show()

