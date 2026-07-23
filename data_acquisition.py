import numpy as np
# import numba as nb
import scipy.linalg as la 
import matplotlib.pyplot as plt 
import os
import pickle
import sys

from system_init import system_init 
from methods import methods
from analysis import pre_analysis, post_analysis

def save_data(path_sub, seed, chi_init, T, kappa, beta, C, mag_elas, pop_link_dict, mu_arr, eival, eivec): 
    path_head = "/home/kuerschner/Documents/Master-Project/data"

    save_path = os.path.join(path_head, path_sub)

    if os.path.isdir(save_path):
        print("error: folder", save_path, " already exists. Routine terminated.")
        sys.exit()

    else: 

        os.makedirs(save_path)


        info = {
                'T': T,
                'kappa': kappa,
                'beta': beta,
                'C' : C,
                'mag_elas' : mag_elas, 
                }

        miscellaneous = {
                'seed' : seed,
                'chi_init': chi_init
                }


        f = open(os.path.join(save_path, 'pop_link_dict'+'.pkl'), 'wb')
        pickle.dump(pop_link_dict, f)

        f = open(os.path.join(save_path, 'mu_arr'+'.pkl'), 'wb')
        pickle.dump(mu_arr, f)

        f = open(os.path.join(save_path, 'eival'+'.pkl'), 'wb')
        pickle.dump(eival, f)

        f = open(os.path.join(save_path, 'eivec'+'.pkl'), 'wb')
        pickle.dump(eivec,  f)

        f = open(os.path.join(save_path, 'info'+'.pkl'), 'wb')
        pickle.dump(info, f)

        f = open(os.path.join(save_path, 'miscellaneous' + '.pkl') , 'wb')
        pickle.dump(miscellaneous, f)

    return 



class data_miner:

    def __init__(self):
        return

    def cond_size_iter(seed, rng, beta, kappa, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras, target_con, target):

        for T in [25, 30, 35, 40, 45, 50, 55]:
                
            i = 1
            found_target = False

            while found_target == False:
                
                print("now doing T=", T, " run number:", i)
                i += 1

                path_sub = os.path.join("fs_extrapol", project_name, f"T={T}")

                init = system_init(T, kappa, rng) 
                link_dict, plaqu_dict, mu_arr, pop_link_dict = init.init_master(chi_init)

                meth = methods(T, beta, kappa)
                mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, free_en_hist, eival, eivec, sc_iter = meth.MF_solver(rng, iter_paras, pre_ana_paras, convergence_paras, link_dict, plaqu_dict, mu_arr, pop_link_dict)

                post_ana = post_analysis(T, kappa, beta, plaqu_dict, pop_link_dict, eival, eivec, mu_hist_dict,  mu_arr, bond_hist_dict, plaqu_hist_dict, sc_iter)                    

                max_iter_cond, sc_iter_max = iter_paras

                if max_iter_cond == False or (max_iter_cond == True and sc_iter != sc_iter_max):

                    if target_con == True:
                        found_target = post_ana.plaqu_dict_check(target)
                
                        if found_target == True:
                            save_data(path_sub, seed, chi_init, T, kappa, beta, pop_link_dict, mu_arr, eival, eivec) 


                    elif target_con == False:
                        found_target = True
                        save_data(path_sub, seed, chi_init, T, kappa, beta, pop_link_dict, mu_arr, eival, eivec)


    def zero_T_iter(seed, rng, T, kappa, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras):

        for beta in [10, 20, 30, 50, 100, 150, 200, 250, 300]:

            print(rf"now doing $\beta$=", beta)

            path_sub = os.path.join("zero_T_extrapol", project_name, f"beta = {beta}")

            init = system_init(T, kappa, rng) 
            link_dict, plaqu_dict, mu_arr, pop_link_dict = init.init_master(chi_init)

            meth = methods(T, beta, kappa)
            mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, free_en_hist, eival, eivec, sc_iter = meth.MF_solver(rng, iter_paras, pre_ana_paras, convergence_paras, link_dict, plaqu_dict, mu_arr, pop_link_dict)

            max_iter_cond, sc_iter_max = iter_paras

            if max_iter_cond == False or (max_iter_cond == True and sc_iter != sc_iter_max):

                save_data(path_sub, seed, chi_init, T, kappa, beta, pop_link_dict, mu_arr, eival, eivec)


    def strain_iter(seed, rng, T, kappa, beta, mag_elas, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras):

        for C in [0, 0.05, ]:

            print(rf"now doing $\beta$=", beta)

            path_sub = os.path.join("zero_T_extrapol", project_name, f"beta = {beta}")

            init = system_init(T, kappa, rng) 
            link_dict, plaqu_dict, mu_arr, pop_link_dict = init.init_master(chi_init)

            meth = methods(T, beta, kappa)
            mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, free_en_hist, eival, eivec, sc_iter = meth.MF_solver(rng, iter_paras, pre_ana_paras, convergence_paras, link_dict, plaqu_dict, mu_arr, pop_link_dict)

            max_iter_cond, sc_iter_max = iter_paras

            if max_iter_cond == False or (max_iter_cond == True and sc_iter != sc_iter_max):

                save_data(path_sub, seed, chi_init, T, kappa, beta, pop_link_dict, mu_arr, eival, eivec)



                
