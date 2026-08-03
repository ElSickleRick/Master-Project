import sys
sys.path.insert(0, '/home/kuerschner/Documents/Master-Project/python')
import numpy as np
import scipy.linalg as la 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider
import matplotlib.patches as ptch
import matplotlib.colors as clr
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import pickle
from system_init import system_init
from analysis import post_analysis

path_head = "/home/kuerschner/Documents/Master-Project/data/mag_elas_var"



def real_space(): 

    bins = 24

    projects = {
                # 'T=10_3007_01' : "T = 10",
                'T=10_3007_01' : "T=10",
                # 'T=10_rot_3007_01' : "T = 10 + rotation",
                }

    fig, ax  = plt.subplots(1,3)
    # ax[0].set_aspect('equal', adjustable='box')
    plt.subplots_adjust(bottom=0.25)
    

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        mag_elas_arr = []
        grid_dict = {}
        pop_link_dict_dict = {}
        eival_dict = {}

        for mag_elas in os.listdir(project_path):

            size_path = os.path.join(project_path, mag_elas)

            with open(os.path.join(size_path, "pop_link_dict.pkl"), "rb") as f:
                pop_link_dict = pickle.load(f)

            with open(os.path.join(size_path, "mu_arr.pkl"), "rb") as f:
                mu_arr = pickle.load(f)

            with open(os.path.join(size_path, "eival.pkl"), "rb") as f:
                eival = pickle.load(f)

            with open(os.path.join(size_path, "eivec.pkl"), "rb") as f:
                eivec = pickle.load(f)

            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)

            T = info["T"]
            kappa = info["kappa"]
            beta = info["beta"]
            C = info["C"]
            mag_elas = info["mag_elas"]
            theta = info["theta"]

            with open(os.path.join(size_path, "miscellaneous.pkl"), "rb") as f:
                miscellaneous = pickle.load(f)
            seed = miscellaneous["seed"]
            chi_init = miscellaneous["chi_init"]
        
            rng = np.random.default_rng(seed) # This is possably questionable but the functions I want to call actually does not need rng       
            sys_init = system_init(T, kappa, rng, C, mag_elas, theta)
            strain_cord_dict = sys_init.strain_cord_gen()
            plaqu_dict = sys_init.plaqu_dict_gen()
            ul_dict = sys_init.link_dict_gen()

            grid = np.empty((0,2))

            for i in range(1, int((T+1)*(T+2)/2+1)):
            
                cords = np.array([strain_cord_dict[i][1]])
                grid = np.append(grid, cords, axis = 0)


            mag_elas_arr.append(mag_elas)
            grid_dict.update({mag_elas : grid})
            pop_link_dict_dict.update({mag_elas : pop_link_dict})
            eival_dict.update({mag_elas : eival})

    cmap = clr.LinearSegmentedColormap.from_list("periodic", ["purple", "blue", "white",  "red", "purple"])
    norm = clr.Normalize(vmin = -np.pi, vmax = np.pi)


    def plot(i):

        variant = "chi" # if "chi" -> abs(chi) ist plotted, if "t" -> hopping amplitude is plotted

        lw_min = 0
        lw_max = 8
        t_min = 0
        t_max = 5
        chi_abs_min = 0
        chi_abs_max = 0.5

        for link in pop_link_dict_dict[mag_elas_arr[i]]:

            s, e, J, chi = pop_link_dict_dict[mag_elas_arr[i]][link]

            x = [grid_dict[mag_elas_arr[i]][s-1][0], grid_dict[mag_elas_arr[i]][e-1][0]]
            y = [grid_dict[mag_elas_arr[i]][s-1][1], grid_dict[mag_elas_arr[i]][e-1][1]]
            chi_abs = np.absolute(chi)

            if chi_abs < 0.001:
                ax[0].plot(x, y, c = 'grey', linestyle = 'dotted', zorder = 3)
            
            else:
                if variant == "chi":
                    lw = (lw_max - lw_min)/(chi_abs_max - chi_abs_min)*(np.absolute(chi)-chi_abs_min) + lw_min

                elif variant == "t":
                    if J*np.absolute(chi)*(1+self.kappa/2-4*self.kappa*np.absolute(chi)**2) > t_max:
                        print("warning: upper t-bound for line thicknes too small")
                    lw = (lw_max - lw_min)/(t_max - t_min)*(J*np.absolute(chi)*(1+self.kappa/2-4*self.kappa*np.absolute(chi)**2-t_min)) + lw_min

                ax[0].plot(x, y, c = 'k', linewidth = lw, zorder = 3)
        
        phase_arr = []

        for plaqu in plaqu_dict:
            orientation, corners = plaqu_dict[plaqu]
            

            if orientation == 'up':
                base = pop_link_dict_dict[mag_elas_arr[i]][str(corners[0]) + str(corners[1])][3]
                right = pop_link_dict_dict[mag_elas_arr[i]][str(corners[1])+str(corners[2])][3]
                left = pop_link_dict_dict[mag_elas_arr[i]][str(corners[0]) + str(corners[2])][3]
                phase = np.angle(base*np.conjugate(left)*right) 


            elif orientation == 'down':
                right = pop_link_dict_dict[mag_elas_arr[i]][str(corners[0]) + str(corners[2])][3]
                top = pop_link_dict_dict[mag_elas_arr[i]][str(corners[1]) + str(corners[2])][3]
                left = pop_link_dict_dict[mag_elas_arr[i]][str(corners[0]) + str(corners[1])][3]
                phase = np.angle(right*np.conjugate(top)*np.conjugate(left))


            
            phase_arr.append(phase)
            color = cmap(norm(phase))
            triangle = ptch.Polygon([grid_dict[mag_elas_arr[i]][int(corners[0]-1)], grid_dict[mag_elas_arr[i]][int(corners[1]-1)], grid_dict[mag_elas_arr[i]][int(corners[2]-1)]], color = color, zorder = 1)
            ax[0].add_patch(triangle)


        ax[1].hist(phase_arr, bins = 25, range =  [np.pi/2 - np.pi/20, np.pi/2 + np.pi/20])
        # ax[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],)
        # ax[1].set_xticklabels([f"-$\pi$", f"$-\pi/2$", "0", f"$\pi/2$", f"$\pi$"])

        ax[2].hist(eival_dict[mag_elas_arr[i]]/np.exp(-mag_elas_arr[i]*C), bins = 24)
        
        return 
    

    mag_elas_current = 0
    plot(0)

    sm = cm.ScalarMappable(cmap = cmap, norm = norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = ax) 
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([f"$-\pi$", f"$-\pi / 2$", "0", f"$\pi / 2$", f"$\pi$"])   

    ax_mag_elas_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    mag_elas_slider = Slider(
            ax = ax_mag_elas_slider,
            label = 'magneto-elastic coupling',
            valmin = 0,
            valmax = len(mag_elas_arr)-1,
            valinit = mag_elas_current,
            valstep = 1
            )

    def mag_elas_update(value):
        i = int(mag_elas_slider.val)
        for axes in ax:
            axes.clear()
        plot(i)
        return

    mag_elas_slider.on_changed(mag_elas_update)
    plt.show()


def J_t():

    bins = 15


    projects = {
                # 'T=10_3007_01' : "T = 10",
                'T=10_3007_01' : "T=10",
                # 'T=10_rot_3007_01' : "T = 10 + rotation",
                }

    fig, ax  = plt.subplots(1,3)  
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        J_dict = {}
        t_dict = {}
        chi_abs_dict = {}
        mag_elas_arr = []

        for mag_elas in os.listdir(project_path):

            size_path = os.path.join(project_path, mag_elas)

            with open(os.path.join(size_path, "pop_link_dict.pkl"), "rb") as f:
                pop_link_dict = pickle.load(f)

            with open(os.path.join(size_path, "mu_arr.pkl"), "rb") as f:
                mu_arr = pickle.load(f)

            with open(os.path.join(size_path, "eival.pkl"), "rb") as f:
                eival = pickle.load(f)

            with open(os.path.join(size_path, "eivec.pkl"), "rb") as f:
                eivec = pickle.load(f)

            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)

            T = info["T"]
            kappa = info["kappa"]
            beta = info["beta"]
            C = info["C"]
            mag_elas = info["mag_elas"]
            theta = info["theta"]

            with open(os.path.join(size_path, "miscellaneous.pkl"), "rb") as f:
                miscellaneous = pickle.load(f)
            seed = miscellaneous["seed"]
            chi_init = miscellaneous["chi_init"]
            
            mag_elas_arr.append(mag_elas)
            J_dict.update({mag_elas : []})
            chi_abs_dict.update({mag_elas : []})
            t_dict.update({mag_elas: []})

            for x in pop_link_dict.keys():
                s, e, J, chi = pop_link_dict[x]

                J_dict[mag_elas].append(J)
                chi_abs_dict[mag_elas].append(np.absolute(chi))
                t_dict[mag_elas].append(np.absolute(J*chi*(1+kappa/2 - 4*kappa*np.absolute(chi)**2)))

    ax[0].hist(J_dict[mag_elas_arr[0]], bins = bins)
    ax[1].hist(chi_abs_dict[mag_elas_arr[0]], bins = bins)
    ax[2].hist(t_dict[mag_elas_arr[0]], bins = bins)
    
    mag_elas_current = 0
    ax_mag_elas_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    mag_elas_slider = Slider(
            ax = ax_mag_elas_slider,
            label = 'magneto-elastic coupling',
            valmin = 0,
            valmax = len(mag_elas_arr)-1,
            valinit = mag_elas_current,
            valstep = 1
            )

    def mag_elas_update(value):
        i = int(mag_elas_slider.val)
        for axes in ax:
            axes.clear()
        ax[0].hist(J_dict[mag_elas_arr[i]], bins = bins)
        ax[1].hist(chi_abs_dict[mag_elas_arr[i]], bins = bins)
        ax[2].hist(t_dict[mag_elas_arr[i]], bins = bins)
        return
       
    mag_elas_slider.on_changed(mag_elas_update)
    plt.show()


def LDOS():

    bins = 25
    width = 0.1


    projects = {
                # 'T=10_3007_01' : "T = 10",
                'T=10_3007_01' : "T=10",
                # 'T=10_rot_3007_01' : "T = 10 + rotation",
                }

    fig, ax  = plt.subplots()
    axins = inset_axes(ax, width="30%", height="30%", loc="upper right")
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        mag_elas_arr = []
        eival_dict = {}
        eivec_avs_dict = {}

        for mag_elas in os.listdir(project_path):

            size_path = os.path.join(project_path, mag_elas)

            with open(os.path.join(size_path, "pop_link_dict.pkl"), "rb") as f:
                pop_link_dict = pickle.load(f)

            with open(os.path.join(size_path, "mu_arr.pkl"), "rb") as f:
                mu_arr = pickle.load(f)

            with open(os.path.join(size_path, "eival.pkl"), "rb") as f:
                eival = pickle.load(f)

            with open(os.path.join(size_path, "eivec.pkl"), "rb") as f:
                eivec = pickle.load(f)

            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)

            T = info["T"]
            kappa = info["kappa"]
            beta = info["beta"]
            C = info["C"]
            mag_elas = info["mag_elas"]
            theta = info["theta"]

            with open(os.path.join(size_path, "miscellaneous.pkl"), "rb") as f:
                miscellaneous = pickle.load(f)
            seed = miscellaneous["seed"]
            chi_init = miscellaneous["chi_init"]
            
            mag_elas_arr.append(mag_elas)
            eivec_avs_dict.update({mag_elas : np.absolute(eivec)**2})
            eival_dict.update({mag_elas: eival})

        
        rng = np.random.default_rng(seed) # This is possably questionable but the functions I want to call actually does not need rng       
        sys_init = system_init(T, kappa, rng, C, mag_elas, theta)
        strain_cord_dict = sys_init.strain_cord_gen()
        ul_dict = sys_init.link_dict_gen()

        grid = np.empty((0,2))

        for i in range(1, int((T+1)*(T+2)/2+1)):
            
            cords = np.array([strain_cord_dict[i][1]])
            grid = np.append(grid, cords, axis = 0)    

    def plot_grid(): 
        for link in ul_dict:

            s, e = ul_dict[link]

            x = [grid[s-1][0],grid[e-1][0]]
            y = [grid[s-1][1], grid[e-1][1]]

            axins.plot(x, y, c = "grey", linewidth = 1, zorder = 2)

    mag_elas_current = 0
    pos_current = 32
    
    ax.bar(eival_dict[mag_elas_current], eivec_avs_dict[mag_elas_current][pos_current, :], width  = width)
    # ax.hist(LDOS_dict[mag_elas_current][pos_current, :], bins = bins)
    plot_grid()

    axins.scatter_artist = axins.scatter(grid[pos_current, 0], grid[pos_current, 1], s=20, color='red')

    
    ax_mag_elas_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    mag_elas_slider = Slider(
            ax = ax_mag_elas_slider,
            label = 'mag_elas',
            valmin = 0,
            valmax = len(mag_elas_arr)-1,
            valinit = mag_elas_current,
            valstep = 1
            )

    ax_pos_slider = plt.axes([0.2, 0.05, 0.6, 0.05])
    pos_slider = Slider(
            ax = ax_pos_slider,
            label = 'position',
            valmin = 0,
            valmax = int((T+1)*(T+2)/2-1),
            valinit = pos_current,
            valstep = 1
            )

    def slider_update(val):

        i = int(mag_elas_slider.val)
        pos  = int(pos_slider.val)

        ax.clear()
        axins.scatter_artist.set_visible(False)
        
        ax.bar(eival_dict[mag_elas_arr[i]], eivec_avs_dict[mag_elas_arr[i]][pos, :], width = width)
        # ax.hist(LDOS_dict[mag_elas_arr[i]][pos, :], bins = bins)
        axins.scatter_artist = axins.scatter(grid[pos, 0], grid[pos, 1], s=20, color='red')


    mag_elas_slider.on_changed(slider_update)
    pos_slider.on_changed(slider_update)

    plt.show()






# real_space_plot_slider()
# J_t_hist_plot()
LDOS()

