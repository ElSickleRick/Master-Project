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

def save_data(path_sub, seed, T, kappa, beta, pop_link_dict, mu_arr, eival, eivec): 
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
                'beta': beta
                }

        miscellaneous = {
                'seed' : seed 
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

    def __init__(self, seed, rng, T, beta, kappa):
        return

    def cond_size_iter(seed, rng, beta, kappa, project_name, chi_init, iter_paras, pre_ana_paras, convergence_paras, target_con, target):

        for T in [30, 35, 40, 45, 50, 55]:
                
            i = 1
            found_target = False

            while found_target == False:
                
                print("now doing T=", T, " run number:", i)
                i += 1

                path_sub = os.path.join(project_name, f"T={T}")

                init = system_init(T, kappa, rng) 
                link_dict, plaqu_dict, mu_arr, pop_link_dict = init.init_master(chi_init)

                meth = methods(T, beta, kappa)
                mu_arr, pop_link_dict, mu_hist_dict, bond_hist_dict, plaqu_hist_dict, free_en_hist, eival, eivec, sc_iter = meth.MF_solver(rng, iter_paras, pre_ana_paras, convergence_paras, link_dict, plaqu_dict, mu_arr, pop_link_dict)

                post_ana = post_analysis(T, kappa, beta, plaqu_dict, pop_link_dict, eival, eivec, mu_hist_dict,  mu_arr, bond_hist_dict, plaqu_hist_dict, sc_iter)                    
                

                if target_con == True:
                    found_target = post_ana.plaqu_dict_check(target)
                
                    if found_target == True:
                        save_data(path_sub, seed, T, kappa, beta, pop_link_dict, mu_arr, eival, eivec) 


                elif target_con == False:
                    found_target = True
                    save_data(path_sub, seed, T, kappa, beta, pop_link_dict, mu_arr, eival, eivec)

                
