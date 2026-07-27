import sys
sys.path.insert(0, '/home/kuerschner/Documents/Master-Project/python')
import numpy as np
import scipy.linalg as la 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider
import matplotlib.patches as ptch
import matplotlib.colors as clr
import matplotlib.cm as cm
import os
import pickle
from system_init import system_init
from analysis import post_analysis

path_head = "/home/kuerschner/Documents/Master-Project/data/strain_var"

def DOS_slider():
    bins = 24

    projects = {
            # 'T=35_2707' : "T=35",
            # 'T=45_2707_01' :  "T=45",
            # 'T=45_rot_2707_01' : "T=45 rotated",
                }

    fig, ax  = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        C_arr = []
        eival_dict = {}

        for C in os.listdir(project_path):

            size_path = os.path.join(project_path, C)

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
        
            C_arr.append(C)
            eival_dict.update({C: eival})

            with open(os.path.join(size_path, "miscellaneous.pkl"), "rb") as f:
                miscellaneous = pickle.load(f)
            seed = miscellaneous["seed"]
            chi_init = miscellaneous["chi_init"]
        
    current = 0
    ax.hist(eival_dict[0], bins = bins)
    ax.set_title(f"C = {current}")

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    slider = Slider(
            ax = ax_slider,
            label = 'C',
            valmin = 0,
            valmax = len(C_arr)-1, 
            valinit = current,
            valstep = 1
            )

    def update(val):
        i = int(slider.val)
        ax.clear()
        ax.hist(eival_dict[C_arr[i]], bins = bins)
        ax.set_title(f"C = {C_arr[i]}")
        fig.canvas.draw_idle()


    slider.on_changed(update)
    plt.show()




def state_real_space_plot():
    
    bins = 24

    projects = {
            'T=45_2707_01' :  "T=45",
            # 'T=45_rot_2707_01' : "T=45 rotated",
                }

    fig, ax  = plt.subplots(1,2)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        C =0.5

        size_path = os.path.join(project_path, f'C = {C}')

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
        sys_init = system_init(T, kappa, rng, C, mag_elas)
        strain_cord_dict = sys_init.strain_cord_gen(theta) 
        
        i = np.argmin(np.absolute(eival))
        weights = eivec[:, i]
       
        colors = []
        cmap = clr.LinearSegmentedColormap.from_list("white_red_blac", [(0, "white"), (0.3, "red"), (1, "black")], gamma=0.75)

        grid = np.empty((0,2))

        for i in range(1, int((T+1)*(T+2)/2+1)):
            
            cords = np.array([strain_cord_dict[i][1]])
            grid = np.append(grid, cords, axis = 0)
            colors.append(cmap(np.absolute(weights[int(i-1)])))

        ax[0].scatter(grid[:,0], grid[:,1], s = 30, c= colors,marker ="o", zorder = 3)


        for link in pop_link_dict:

            s, e, J, chi = pop_link_dict[link]

            x = [grid[s-1][0], grid[e-1][0]]
            y = [grid[s-1][1], grid[e-1][1]]


            ax[0].plot(x, y, c = 'grey', linewidth = 1, zorder = 2)

        ax[1].hist(eival)
        
        plt.show()


def state_real_space_plot_slider():
    
    bins = 24

    projects = {
            'T=45_2707_01' :  "T=45",
            # 'T=45_rot_2707_01' : "T=45 rotated",
                }

    fig, ax  = plt.subplots(1,2)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        C_arr = []
        eival_dict = {}
        eivec_dict = {}
        grid_dict = {}

        for C in os.listdir(project_path):

            size_path = os.path.join(project_path, C)

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
            ul_dict = sys_init.link_dict_gen()

            grid = np.empty((0,2))

            for i in range(1, int((T+1)*(T+2)/2+1)):
            
                cords = np.array([strain_cord_dict[i][1]])
                grid = np.append(grid, cords, axis = 0)


            C_arr.append(C)
            eival_dict.update({C : eival})
            eivec_dict.update({C : eivec})
            grid_dict.update({C : grid})
        
     
    colors = []
    cmap = clr.LinearSegmentedColormap.from_list("white_red_blac", [(0, "white"), (0.3, "red"), (1, "black")], gamma=0.75)

    C_current = 0
    eigen_current = 0

    ax[0].scatter(grid_dict[C_arr[0]][:,0], grid_dict[C_arr[0]][:,1] , s = 20, c = [cmap(np.absolute(x)) for x in eivec_dict[C_arr[0]][:,0]], marker = "o", zorder = 3)

    for link in ul_dict:

        s, e = ul_dict[link]

        x = [grid_dict[C_arr[0]][s-1][0],grid_dict[C_arr[0]][e-1][0]]
        y = [grid_dict[C_arr[0]][s-1][1], grid_dict[C_arr[0]][e-1][1]]

        ax[0].plot(x, y, c = "grey", linewidth = 1, zorder = 2)

    ax[1].hist(eival_dict[C_arr[0]], bins)
    ax[1].axvline(eival_dict[0][0], color = "red", zorder = 3)

    ax_C_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    C_slider = Slider(
            ax = ax_C_slider,
            label = 'C',
            valmin = 0,
            valmax = len(C_arr)-1,
            valinit = C_current,
            valstep = 1
            )

    ax_eigen_slider = plt.axes([0.2, 0.05, 0.6, 0.05])
    eigen_slider = Slider(
            ax = ax_eigen_slider,
            label = 'E',
            valmin = 0,
            valmax = int((T+1)*(T+2)/2),
            valinit = eigen_current,
            valstep = 1
            )
            

    def C_update(val):
        i = int(C_slider.val)
        e = int(eigen_slider.val)

        ax[0].clear()
        ax[0].scatter(grid_dict[C_arr[i]][:,0], grid_dict[C_arr[i]][:,1] , s = 20, c = [cmap(np.absolute(x)) for x in eivec_dict[C_arr[i]][:,e]], marker = "o", zorder = 3) 

        ax[1].clear()
        ax[1].hist(eival_dict[C_arr[i]], bins)
        ax[1].axvline(eival_dict[C_arr[i]][e], color = "red", zorder = 3)        
        for link in ul_dict:

            s, e = ul_dict[link]

            x = [grid_dict[C_arr[i]][s-1][0], grid_dict[C_arr[i]][e-1][0]]
            y = [grid_dict[C_arr[i]][s-1][1], grid_dict[C_arr[i]][e-1][1]]

            ax[0].plot(x, y, c = "grey", linewidth = 1, zorder = 2)

    def eigen_update(val):
        i = int(C_slider.val)
        e = int(eigen_slider.val)

        for scatter in list(ax[0].collections):
            scatter.remove()

        ax[0].scatter(grid_dict[C_arr[i]][:,0], grid_dict[C_arr[i]][:,1] , s = 20, c = [cmap(np.absolute(x)) for x in eivec_dict[C_arr[i]][:,e]], marker = "o", zorder = 3)

        for lines in list(ax[1].lines):
            lines.remove()
        ax[1].axvline(eival_dict[C_arr[i]][e], color = "red", zorder = 3)

    C_slider.on_changed(C_update)
    eigen_slider.on_changed(eigen_update)

    plt.show()



# DOS_slider()
# gap_state_real_space_plot()
state_real_space_plot_slider()
