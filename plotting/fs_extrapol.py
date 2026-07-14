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


    mu_part = -np.sum(mu_arr)

    return 2*(-(2/beta)*F + mu_part)/((T+1)*(T+2))


def free_en_calc(T, beta, kappa, eival, mu_arr, pop_link_dict):
    '''
    calculates the free energy desnsity of a given mean field solution

    Returns:
    --------
    free_en: float 
            free energy  
    '''


    F = 0
    for x in eival:
        if x < 0:
            F += 2*x-(2/beta)*np.log(1+np.exp(beta*x))
        elif x > 0:
            F += -(2/beta)*np.log(1+np.exp(-beta*x))

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


path_head = "/home/kuerschner/Documents/Master-Project/data/fs_extrapol"


projects = {
        "down_0207_01": "down",
        # "up_0207_01": "up",
        "zero_0307_01": "zero", # without randomness
        # "pi_0307_01" : "pi", # without randomness
        # "pi_half_1407_01": "pi/2",
        "minus_pi_half_1407_01": "-pi/2",
        }


# [mu, chi_abs, free_energy] ;) 
tl = {
        "down" : [0, 0.19754, -1.2206],
        "up" : [0, 0.19754, -1.2206],
        "zero" : [-0.67574, 0.164712, -0.888302],
        "pi" : [0.67574, 0.164712, -0.888302],
        "pi/2": [0, 0.200169, -1.24978],
        "-pi/2": [0, 0.200169, -1.24978],
        }


x_max = 0.06
fig, ax = plt.subplots(1,3)
fit_range = np.linspace(0, x_max, 1000)


mu_title = chi_abs_title = free_energy_title = f"deviation from analytical value:"

for project in projects:

    project_path = os.path.join(path_head, project)
    

    mu_tl, chi_abs_tl, free_energy_tl = tl[projects[project]]

    T_arr = []
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
       
        N = ((T+1)*(T+2))/2 # # sites, only needed for standard deviation of mean! 
        T_arr.append(int(T))

        mu_mean.append(np.mean(mu_arr))
        mu_stdm.append(np.std(mu_arr, ddof = 1)/(np.sqrt(N)))

        chi_abs_arr = []

        for link in pop_link_dict:
            chi_abs_arr.append(np.absolute(pop_link_dict[link][3]))

        chi_abs_mean.append(np.mean(chi_abs_arr))
        chi_abs_stdm.append(np.std(chi_abs_arr, ddof = 1)/(np.sqrt(N)))

        fe_di = np.array([lin_fe_di(beta*x) for x in eival])

        free_energy_arr.append(free_en_calc(T, beta, kappa, eival, mu_arr, pop_link_dict))



    inv_T = [1/s for s in T_arr]

    even_mask = (np.array(T_arr) % 2 == 0)
    mu_fit_even = np.polyfit(np.array(inv_T)[even_mask], np.array(mu_mean)[even_mask], 1, w = [1/s for s in np.array(mu_stdm)[even_mask]])

    ax[0].errorbar(np.array(inv_T)[even_mask], np.array(mu_mean)[even_mask], yerr = np.array(mu_stdm)[even_mask], fmt='x', label = str(projects[project]) + "even")
    ax[0].plot(fit_range, [fit_fun(x,mu_fit_even) for x in fit_range], label = f"fit {projects[project]}" + "even")
    ax[0].hlines(mu_tl, 0, x_max, linestyle = "--", color = 'k')
    mu_title += f" \n {projects[project]} even: {mu_fit_even[len(mu_fit_even)-1] - mu_tl} J"

    odd_mask = np.logical_not(even_mask)

    mu_fit_odd = np.polyfit(np.array(inv_T)[odd_mask], np.array(mu_mean)[odd_mask], 1, w = [1/s for s in np.array(mu_stdm)[odd_mask]])

    ax[0].errorbar(np.array(inv_T)[odd_mask], np.array(mu_mean)[odd_mask], yerr = np.array(mu_stdm)[odd_mask], fmt='x', label = f"{projects[project]}" + "odd")
    ax[0].plot(fit_range, [fit_fun(x,mu_fit_odd) for x in fit_range], label = f"fit {projects[project]}" + "odd")
    mu_title += f" \n {projects[project]} odd: {mu_fit_odd[-1] - mu_tl} J"


    chi_abs_fit = np.polyfit(inv_T, chi_abs_mean, 1, w = [1/s for s in chi_abs_stdm])

    ax[1].errorbar(inv_T, chi_abs_mean, yerr = chi_abs_stdm, fmt='x', label = str(projects[project]))
    ax[1].plot(fit_range, [fit_fun(x,chi_abs_fit) for x in fit_range], label = f"fit {projects[project]}")
    ax[1].hlines(chi_abs_tl, 0, x_max, linestyle = "--", color = 'k')
    chi_abs_title += f" \n {projects[project]}: {chi_abs_fit[-1] - chi_abs_tl}"

    free_energy_fit = np.polyfit(inv_T, free_energy_arr, 1)

    ax[2].scatter(inv_T, free_energy_arr, label = str(projects[project]))
    ax[2].plot(fit_range, [fit_fun(x,free_energy_fit) for x in fit_range], label = f"fit {projects[project]}")
    ax[2].hlines(free_energy_tl, 0, x_max, linestyle = "--", color = 'k')
    free_energy_title =  free_energy_title + f" \n {projects[project]}: {free_energy_fit[-1] - free_energy_tl} J"

    ax[0].title.set_text(mu_title)
    ax[1].title.set_text(chi_abs_title)
    ax[2].title.set_text(free_energy_title)
    
    ax[0].set_ylabel("$\mu$ in units of J", fontsize = "x-large")
    ax[1].set_ylabel("$|\chi|$", fontsize = "x-large")
    ax[2].set_ylabel("free energy in unts of J", fontsize = "x-large")
    
fig.suptitle("finite size extrapolation")

for axes in ax:
    axes.set_xlabel(r"(linear system size)$^{-1}$")
    axes.set_xlim([0, x_max])
    axes.legend()



plt.show()

