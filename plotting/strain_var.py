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
    bins = 24 # nu,ber of bins in the histogram
    categories = [0.9, 0.95, 0.99]

    def edge_projection(T, eivec):

        """
        returns AVS of projection of eigenvectors onto edge of sample
        
        OUTPUTS:
        -------
        projection: 1D array (len: # sites)
                    AVS of projcetion of eigenvectors onto edge of sample, ordered ascendingly wrt eigenvalues
        """

        projector = np.zeros(int((T+1)*(T+2)/2))

        # keep in mind that site counting starts at 1 but indexing starts at 0
        for i in range(0, T +1): # run over base of the traingle
            projector[int((i+1)*i/2)] = 1
        
        for i in range(2, T+2): # run over right side of triangle 
            projector[int(T*(T+1)/2+i-1)] = 1

        for i in range(2, T +1): # run over left side of triangle
            projector[int(i*(i+1)/2 -1)] = 1

        projection = np.transpose(projector) @ (np.absolute(eivec)**2)

        return projection

    projects = {
            'T=45_2707_01' :  "T=45",
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
            
            projection = edge_projection(T, eivec)
            
            eival_dict.update({C : np.array([eival[projection < categories[0]]])})
            
            for i in(0, len(categories)-2):

                states = np.logical_and( projection > categories[i], projection <= categories[i+1])
                np.append(eival_dict[C], [states], axis = 0)

            np.append(eival_dict[C], [eival[projection > categories[-1]]], axis = 0)
            print(np.shape(eival_dict[C]))

            with open(os.path.join(size_path, "miscellaneous.pkl"), "rb") as f:
                miscellaneous = pickle.load(f)
            seed = miscellaneous["seed"]
            chi_init = miscellaneous["chi_init"]

    
    
    current = 0
    ax.hist(eival_dict[C_arr[0]], bins = bins, stacked = True)
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
        ax.hist(eival_dict[C_arr[i]], bins = bins, stacked = True)
        ax.set_title(f"C = {C_arr[i]}")
        fig.canvas.draw_idle()


    slider.on_changed(update)
    plt.show()

def real_space_plot_slider(): 

    bins = 24

    projects = {
            'T=45_2707_01' :  "T=45",
            # 'T=45_rot_2707_01' : "T=45 rotated",
                }

    fig, ax  = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        C_arr = []
        grid_dict = {}
        pop_link_dict_dict = {}

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
            plaqu_dict = sys_init.plaqu_dict_gen()
            ul_dict = sys_init.link_dict_gen()

            grid = np.empty((0,2))

            for i in range(1, int((T+1)*(T+2)/2+1)):
            
                cords = np.array([strain_cord_dict[i][1]])
                grid = np.append(grid, cords, axis = 0)


            C_arr.append(C)
            grid_dict.update({C : grid})
            pop_link_dict_dict.update({C : pop_link_dict})
    
    def real_space_plot(i):

        variant = "chi" # if "chi" -> abs(chi) ist plotted, if "t" -> hopping amplitude is plotted

        lw_min = 0
        lw_max = 8
        t_min = 0
        t_max = 5
        chi_abs_min = 0
        chi_abs_max = 0.5

        for link in pop_link_dict_dict[C_arr[i]]:

            s, e, J, chi = pop_link_dict_dict[C_arr[i]][link]

            x = [grid_dict[C_arr[i]][s-1][0], grid_dict[C_arr[i]][e-1][0]]
            y = [grid_dict[C_arr[i]][s-1][1], grid_dict[C_arr[i]][e-1][1]]
            chi_abs = np.absolute(chi)

            if chi_abs < 0.001:
                ax.plot(x, y, c = 'grey', linestyle = 'dotted', zorder = 3)
            
            else:
                if variant == "chi":
                    lw = (lw_max - lw_min)/(chi_abs_max - chi_abs_min)*(np.absolute(chi)-chi_abs_min) + lw_min

                elif variant == "t":
                    if J*np.absolute(chi)*(1+self.kappa/2-4*self.kappa*np.absolute(chi)**2) > t_max:
                        print("warning: upper t-bound for line thicknes too small")
                    lw = (lw_max - lw_min)/(t_max - t_min)*(J*np.absolute(chi)*(1+self.kappa/2-4*self.kappa*np.absolute(chi)**2-t_min)) + lw_min

                ax.plot(x, y, c = 'k', linewidth = lw, zorder = 3)
        
        cmap = clr.LinearSegmentedColormap.from_list("periodic", ["purple", "blue", "white",  "red", "purple"])
        norm = clr.Normalize(vmin = -np.pi, vmax = np.pi)

        for plaqu in plaqu_dict:
            orientation, corners = plaqu_dict[plaqu]
            

            if orientation == 'up':
                base = pop_link_dict_dict[C_arr[i]][str(corners[0]) + str(corners[1])][3]
                right = pop_link_dict_dict[C_arr[i]][str(corners[1])+str(corners[2])][3]
                left = pop_link_dict_dict[C_arr[i]][str(corners[0]) + str(corners[2])][3]
                phase = np.angle(base*np.conjugate(left)*right) 


            elif orientation == 'down':
                right = pop_link_dict_dict[C_arr[i]][str(corners[0]) + str(corners[2])][3]
                top = pop_link_dict_dict[C_arr[i]][str(corners[1]) + str(corners[2])][3]
                left = pop_link_dict_dict[C_arr[i]][str(corners[0]) + str(corners[1])][3]
                phase = np.angle(right*np.conjugate(top)*np.conjugate(left))

            
            color = cmap(norm(phase))
            triangle = ptch.Polygon([grid_dict[C_arr[i]][int(corners[0]-1)], grid_dict[C_arr[i]][int(corners[1]-1)], grid_dict[C_arr[i]][int(corners[2]-1)]], color = color, zorder = 1)
            ax.add_patch(triangle)
        return 
    

    C_current = 0
    real_space_plot(0)
   

    ax_C_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    C_slider = Slider(
            ax = ax_C_slider,
            label = 'C',
            valmin = 0,
            valmax = len(C_arr)-1,
            valinit = C_current,
            valstep = 1
            )

    def C_update(value):
        i = int(C_slider.val)
        ax.clear()
        real_space_plot(i)
        return

    C_slider.on_changed(C_update)
    plt.show()





def state_real_space_plot_slider():
    
    bins = 24

    projects = {
            # 'T=45_2707_01' :  "T=45",
            'T=45_rot_2707_01' : "T=45 rotated",
                }

    fig, ax  = plt.subplots(1,2)
    plt.subplots_adjust(bottom=0.25)


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
    cmap = clr.LinearSegmentedColormap.from_list("white_red_blac", [(0, "white"), (1/9, "red"), (1, "black")], gamma=0.75)

    C_current = 0
    eigen_current = 0

    ax[0].scatter(grid_dict[C_arr[0]][:,0], grid_dict[C_arr[0]][:,1] , s = 20, c = [cmap(np.absolute(x)**2) for x in eivec_dict[C_arr[0]][:,0]], marker = "o", zorder = 3)

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
        ax[0].scatter(grid_dict[C_arr[i]][:,0], grid_dict[C_arr[i]][:,1] , s = 20, c = [cmap(np.absolute(x)**2) for x in eivec_dict[C_arr[i]][:,e]], marker = "o", zorder = 3) 

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

        ax[0].scatter(grid_dict[C_arr[i]][:,0], grid_dict[C_arr[i]][:,1] , s = 20, c = [cmap(np.absolute(x)**2) for x in eivec_dict[C_arr[i]][:,e]], marker = "o", zorder = 3)

        for lines in list(ax[1].lines):
            lines.remove()
        ax[1].axvline(eival_dict[C_arr[i]][e], color = "red", zorder = 3)

    C_slider.on_changed(C_update)
    eigen_slider.on_changed(eigen_update)

    plt.show()



DOS_slider()
# gap_state_real_space_plot()
# real_space_plot_slider()
