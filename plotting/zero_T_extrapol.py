import numpy as np
# import numba as nb
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import os
import pickle




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


def fit_fun(x, a):

    y = 0
    m = len(a)
        
    for i in range(0, m):

        y += a[i]*(x**(m-i-1))

    return y 


x_max = 0.15
fig, ax = plt.subplots(1,3)
fit_range = np.linspace(0, x_max, 1000)
path_head = "/home/kuerschner/Documents/Master-Project/data/zero_T_extrapol"


projects = {
        "up_T=25_0207_01" : "T=25 (up)",
        "up_T=30_0207_01" : "T=30 (up)",
        "up_T=35_0207_01" : "T=35 (up)",
        "up_T=40_0207_01" : "T=40 (up)"
            }

for project in projects: 
   
    project_path = os.path.join(path_head, project)

    temp = []
    mu_mean = []
    mu_stdm = []
    chi_abs_mean = []
    chi_abs_stdm = []
    free_energy_arr = []

    for inv_temp in os.listdir(project_path):

        size_path = os.path.join(project_path, inv_temp)
        
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

        N = ((T+1)*(T+2))/2 # # sites, only needed for standard deviation of mean! 
        temp.append(1/beta)

        mu_mean.append(np.mean(mu_arr))
        mu_stdm.append(np.std(mu_arr, ddof = 1)/(np.sqrt(N)))

        chi_abs_arr = []

        for link in pop_link_dict:
            chi_abs_arr.append(np.absolute(pop_link_dict[link][3]))

        chi_abs_mean.append(np.mean(chi_abs_arr))
        chi_abs_stdm.append(np.std(chi_abs_arr, ddof = 1)/(np.sqrt(N)))
        
        free_energy_arr.append(free_en_calc(T, beta, kappa, eival, mu_arr, pop_link_dict))

    mu_fit = np.polyfit(temp, mu_mean, 3, w = [1/s for s in mu_stdm])

    ax[0].errorbar(temp, mu_mean, yerr = mu_stdm, fmt = 'x', ecolor = 'grey', elinewidth = 0.5, capsize = 2, label = str(projects[project]))
    #ax[0].plot( fit_range, [fit_fun(x, mu_fit) for x in fit_range], label = f" fit {projects[project]}")

    chi_fit = np.polyfit(temp, chi_abs_mean, 2, w = [1/s for s in chi_abs_stdm])

    ax[1].errorbar(temp, chi_abs_mean, yerr = chi_abs_stdm, fmt = 'x', ecolor = 'grey', elinewidth = 0.5, capsize = 2, label = str(projects[project]))
    #ax[1].plot( fit_range, [fit_fun(x, chi_fit) for x in fit_range], label = f" fit {projects[project]}")
    
    temp = np.array(temp) 
    free_energy_arr = np.array(free_energy_arr)    
    inf_mask = np.logical_and(free_energy_arr != -np.inf, free_energy_arr != np.inf)
    free_en_fit = np.polyfit(temp[inf_mask], free_energy_arr[inf_mask], 2)

    ax[2].scatter(temp[inf_mask], free_energy_arr[inf_mask], label = str(projects[project]))
    ax[2].plot( fit_range, [fit_fun(x, free_en_fit) for x in fit_range], label = f" fit {projects[project]}") 

    ax[0].set_ylabel("$\mu$", fontsize = "x-large")
    ax[1].set_ylabel("$|\chi|$", fontsize = "x-large")
    ax[2].set_ylabel("free energy", fontsize = "x-large")
    
fig.suptitle("zero temperature extrapolation")

for axes in ax:
    axes.set_xlabel("T")
    axes.set_xlim([0, x_max])
    axes.legend() 

plt.show()



