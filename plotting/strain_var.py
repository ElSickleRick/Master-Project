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

path_head = "/home/kuerschner/Documents/Master-Project/data/strain_var"

def sort(arr):

    sorted_arr = np.array(np.sort(arr), dtype = object)

    zeros = np.where(sorted_arr == 0)[0]
    for i in zeros:
        sorted_arr[i] = int(0)
    
    return sorted_arr
    


def DOS():
    bins = 50 #24 # nmber of bins in the histogram

    fig, ax  = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
        
        C_arr = []
        eival_dict = {}
        mu_arr_dict = {}


        for path in os.listdir(project_path):

            size_path = os.path.join(project_path, path)
            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)  

            C = info["C"]
            C_arr.append(C)

        C_arr = sort(C_arr)

        
        for C in C_arr:

            size_path = os.path.join(project_path, f"C = {C}")

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

            
            eival_dict.update({C : eival})
            mu_arr_dict.update({C : mu_arr})


    
    
    current = 0
    ax.hist(eival_dict[C_arr[0]], bins = bins, fc = "blue", ec = "black")
    ax.vlines(np.mean(mu_arr_dict[C_arr[current]]), 0, 100, color = "red")

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    slider = Slider(
            ax = ax_slider,
            label = 'C',
            valmin = 0,
            valmax = len(C_arr)-1, 
            valinit = current,
            valstep = 1
            )
    slider.valtext.set_text(C_arr[current])

    ax.set_title(f"DOS of {project}", size = "x-large")
 
    params = (
    rf"T = {T}" "\n"
    rf"$\kappa$ = {kappa}" "\n"
    rf"$\beta$ = {beta}" "\n"
    rf"me-coupling = {mag_elas}"
    )

    ax.text(0.02, 0.98, params,transform=ax.transAxes,va="top",ha="left",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),)


    def update(val):
        i = int(slider.val)
        for container in ax.containers:
            container.remove()
        for lines in ax.collections:
            lines.remove()
        ax.vlines(np.mean(mu_arr_dict[C_arr[i]]), 0, 100, color = "red")
        ax.hist(eival_dict[C_arr[i]], bins = bins, fc = "blue", ec = "black")
        fig.canvas.draw_idle()

        slider.valtext.set_text(C_arr[i])

    slider.on_changed(update)
    plt.show()

def LDOS():

    bins = 120 #85 # 65

    fig, ax  = plt.subplots()
    axins = inset_axes(ax, width="30%", height="30%", loc="upper right")
    axins.set_aspect('equal','datalim')
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        C_arr = []
        eival_dict = {}
        eivec_avs_dict = {}
        grid_dict = {}
        mu_arr_dict = {}

        for path in os.listdir(project_path):

            size_path = os.path.join(project_path, path)
            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)  

            C = info["C"]
            C_arr.append(C)

        C_arr = sort(C_arr)

        
        for C in C_arr:

            size_path = os.path.join(project_path, f"C = {C}")
       
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
            chi_init, elas_variant, chi_noise_scale, mu_noise_scale = miscellaneous["init paras"]
            

            eivec_avs_dict.update({C : np.absolute(eivec)**2})
            eival_dict.update({C : eival})
            mu_arr_dict.update({C : mu_arr})
        
            rng = np.random.default_rng(seed) # This is possably questionable but the functions I want to call actually does not need rng       
            sys_init = system_init(T, kappa, rng, C, mag_elas, theta, elas_variant)
            strain_cord_dict = sys_init.strain_cord_gen()
            ul_dict = sys_init.link_dict_gen()

            grid = np.empty((0,2))

            for i in range(1, int((T+1)*(T+2)/2+1)):
            
                cords = np.array([strain_cord_dict[i][1]])
                grid = np.append(grid, cords, axis = 0) 

            grid_dict.update({ C : grid})

    """copied from system_init:"""
                
    s = int((T+1)*(T+2)/2) # # sites
 
    symmetrie_dict = {}

    for i in range(1, s+1):

        c = int(np.ceil(-1/2 + np.sqrt(-3/4 +2*i))) # caclulate column number c
        p = int(i-c*(c-1)/2) # calculate position p inside column c
           
        symmetrie_points = {i}

        c_new = int(-p + 2 + T)
        p_new = int(c - p + 1)
        symmetrie_points.add(int(c_new*(c_new-1)/2+p_new))
           
        c_new = int(-c + p + 1 + T)
        p_new = int(-c + 2 + T)
        symmetrie_points.add(int(c_new*(c_new-1)/2+p_new))
        
        c_new = int(-c + p + 1 + T)
        p_new = int(p)
        symmetrie_points.add(int(c_new*(c_new-1)/2+p_new))
     
        c_new = int(-p + 2 + T)
        p_new = int(-c + 2 + T)
        symmetrie_points.add(int(c_new*(c_new-1)/2+p_new))

        c_new = int(c)
        p_new = int(c - p +1)
        symmetrie_points.add(int(c_new*(c_new-1)/2+p_new))


        symmetrie_dict[i] = sorted(symmetrie_points)


    def plot_grid(i): 
        for link in ul_dict:

            s, e = ul_dict[link]

            x = [grid_dict[C_arr[i]][s-1][0],grid_dict[C_arr[i]][e-1][0]]
            y = [grid_dict[C_arr[i]][s-1][1], grid_dict[C_arr[i]][e-1][1]]

            axins.plot(x, y, c = "grey", linewidth = 1, zorder = 1)

    C_current = 0
    pos_current = int(np.floor(2*T/3)*(np.floor(2*T/3)-1)/2 + np.ceil(T/3))
    sym_current = 1
    
    ax.hist(eival_dict[C_current], bins = bins, weights = eivec_avs_dict[C_current][pos_current-1, :], ec = 'black', fc = 'green')
    ax.vlines(mu_arr_dict[C_arr[C_current]][pos_current- 1], 0, 0.05, color = "red")
    plot_grid(C_current)

    sym_dots = axins.scatter([grid_dict[C_arr[C_current]][i-1, 0] for i in symmetrie_dict[pos_current]], [grid_dict[C_arr[C_current]][i-1, 1] for i in symmetrie_dict[pos_current]], s = 20, color = "blue", zorder = 2)
    pos_dot = axins.scatter(grid_dict[C_arr[C_current]][pos_current-1, 0], grid_dict[C_arr[C_current]][pos_current-1, 1], s=20, color='red', zorder = 3)


    ax.set_xlim(np.min(eival_dict[C_arr[C_current]]) - 0.1 , np.max(eival_dict[C_arr[C_current]]) + 0.1)

    
    ax_C_slider = plt.axes([0.2, 0.15, 0.6, 0.05])
    C_slider = Slider(ax = ax_C_slider, label = 'C', valmin = 0, valmax = len(C_arr)-1, valinit = C_current, valstep = 1)

    ax_pos_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    pos_slider = Slider(ax = ax_pos_slider, label = 'position', valmin = 1, valmax = int((T+1)*(T+2)/2), valinit = pos_current, valstep = 1)

    ax_sym_slider = plt.axes([0.2, 0.05, 0.6, 0.05])
    sym_slider = Slider(ax = ax_sym_slider, label = 'sym', valmin = 1, valmax = len(symmetrie_dict[pos_current]), valinit = sym_current, valstep = 1)
    
    C_slider.valtext.set_text(C_arr[C_current])

    ax.set_title(f"LDOS of {project}", size = "x-large")
 
    params = (
    rf"T = {T} ({int((T+1)*(T+2)/2)} sites)" "\n"
    rf"$\kappa$ = {kappa}" "\n"
    rf"$\beta$ = {beta}" "\n"
    rf"me-coupling = {mag_elas}"
    )

    ax.text(0.02, 0.98, params,transform=ax.transAxes,va="top",ha="left",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),)

    def C_update(val):

        nonlocal sym_dots
        nonlocal pos_dot

        i = int(C_slider.val)
        pos  = int(pos_slider.val)

        sym_slider.valmax = len(symmetrie_dict[pos])
        sym_slider.ax.set_xlim(sym_slider.valmin, sym_slider.valmax)
        sym_slider.eventson = False
        sym_slider.set_val(1)
        sym_slider.eventson = True

        for container in ax.containers:
            container.remove()
        for col in ax.collections:
            col.remove()
        axins.clear()
       
        plot_grid(i)
        ax.hist(eival_dict[C_arr[i]], bins = bins, weights = eivec_avs_dict[C_arr[i]][pos-1, :], ec = 'black', fc = 'green')
        ax.vlines(mu_arr_dict[C_arr[i]][pos-1], 0, 0.05, color = "red")

        sym_dots = axins.scatter([grid_dict[C_arr[i]][j-1, 0] for j in symmetrie_dict[pos]], [grid_dict[C_arr[i]][j-1, 1] for j in symmetrie_dict[pos]], s = 20, color = "blue", zorder = 2)
        pos_dot = axins.scatter(grid_dict[C_arr[i]][pos-1, 0], grid_dict[C_arr[i]][pos-1, 1], s=20, color='red', zorder = 3)
        
        ax.set_xlim(np.min(eival_dict[C_arr[i]]) - 0.1 , np.max(eival_dict[C_arr[i]]) + 0.1)
        C_slider.valtext.set_text(C_arr[i])

    def pos_update(val):

        i = int(C_slider.val)
        pos  = int(pos_slider.val)

        sym_slider.valmax = len(symmetrie_dict[pos])
        sym_slider.ax.set_xlim(sym_slider.valmin, sym_slider.valmax)        
        sym_slider.eventson = False
        sym_slider.set_val(1)
        sym_slider.eventson = True


        for container in ax.containers:
            container.remove()
        for col in ax.collections:
            col.remove()

        ax.hist(eival_dict[C_arr[i]], bins = bins, weights = eivec_avs_dict[C_arr[i]][pos-1, :], ec = 'black', fc = 'green')
        ax.vlines(mu_arr_dict[C_arr[i]][pos-1], 0, 0.05, color = "red")
        pos_dot.set_offsets([grid_dict[C_arr[i]][pos-1, 0], grid_dict[C_arr[i]][pos-1, 1]])
        sym_dots.set_offsets([(grid_dict[C_arr[i]][j-1, 0], grid_dict[C_arr[i]][j-1, 1]) for j in symmetrie_dict[pos]])
        ax.set_xlim(np.min(eival_dict[C_arr[i]]) - 0.1 , np.max(eival_dict[C_arr[i]]) + 0.1)

    def sym_update(val):

        i = int(C_slider.val)
        pos  = int(pos_slider.val)
        sym = int(sym_slider.val)

       
        for container in ax.containers:
            container.remove()
        for col in ax.collections:
            col.remove()

        ax.hist(eival_dict[C_arr[i]], bins = bins, weights = eivec_avs_dict[C_arr[i]][symmetrie_dict[pos][sym-1]-1, :], ec = 'black', fc = 'green')
        ax.vlines(mu_arr_dict[C_arr[i]][pos-1], 0, 0.05, color = "red")
        pos_dot.set_offsets([grid_dict[C_arr[i]][symmetrie_dict[pos][sym-1]-1, 0], grid_dict[C_arr[i]][symmetrie_dict[pos][sym-1]-1, 1]])

        ax.set_xlim(np.min(eival_dict[C_arr[i]]) - 0.1 , np.max(eival_dict[C_arr[i]]) + 0.1)

    C_slider.on_changed(C_update)
    pos_slider.on_changed(pos_update)
    sym_slider.on_changed(sym_update)

    plt.show()

def real_space(mode): 

    bins = 24

    fig, ax  = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        C_arr = []
        grid_dict = {}
        pop_link_dict_dict = {}

        for path in os.listdir(project_path):

            size_path = os.path.join(project_path, path)
            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)  

            C = info["C"]
            C_arr.append(C)

        C_arr = sort(C_arr)

        
        for C in C_arr:

            size_path = os.path.join(project_path, f"C = {C}")

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
            chi_init, elas_variant, chi_noise_scale, mu_noise_scale = miscellaneous["init paras"]
        
            rng = np.random.default_rng(seed) # This is possably questionable but the functions I want to call actually does not need rng       
            sys_init = system_init(T, kappa, rng, C, mag_elas, theta, elas_variant)
            strain_cord_dict = sys_init.strain_cord_gen()
            plaqu_dict = sys_init.plaqu_dict_gen()
            ul_dict = sys_init.link_dict_gen()

            grid = np.empty((0,2))

            for i in range(1, int((T+1)*(T+2)/2+1)):
            
                cords = np.array([strain_cord_dict[i][1]])
                grid = np.append(grid, cords, axis = 0)


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
        


                
        if mode == 'flux':
            cmap = clr.LinearSegmentedColormap.from_list("periodic", ["purple", "blue", "white",  "red", "purple"])
            norm = clr.Normalize(vmin = -np.pi, vmax = np.pi)
            
        elif mode == 'pi/2':
            cmap = clr.LinearSegmentedColormap.from_list("green_pourple", ["green", "white",  "purple"])
            norm = clr.Normalize(vmin = np.pi/2 - 0.001, vmax = np.pi/2 + 0.001)

        elif mode == '-pi/2':
            cmap = clr.LinearSegmentedColormap.from_list("green_pourple", ["green", "white",  "purple"])
            norm = clr.Normalize(vmin = -np.pi/2 - 0.001, vmax = -np.pi/2 + 0.001)


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
        return norm, cmap 
    

    C_current = 0
    norm, cmap = real_space_plot(0)


    sm = cm.ScalarMappable(norm = norm, cmap = cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = ax) 
   

    ax_C_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    C_slider = Slider(
            ax = ax_C_slider,
            label = 'C',
            valmin = 0,
            valmax = len(C_arr)-1,
            valinit = C_current,
            valstep = 1
            )

    C_slider.valtext.set_text(C_arr[C_current])

    ax.set_title(f"{project}", size = "x-large")
 
    params = (
    rf"T = {T}" "\n"
    rf"$\kappa$ = {kappa}" "\n"
    rf"$\beta$ = {beta}" "\n"
    rf"me-coupling = {mag_elas}"
    )

    ax.text(0.02, 0.98, params,transform=ax.transAxes,va="top",ha="left",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def C_update(value):
        i = int(C_slider.val)
        ax.clear()
        real_space_plot(i)
        
        ax.text(0.02, 0.98, params,transform=ax.transAxes,va="top",ha="left",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        C_slider.valtext.set_text(C_arr[i])
        ax.set_title(f"{project}", size = "x-large")
        return

    C_slider.on_changed(C_update) 
    plt.show()





def localization_real_space():
    
    bins = 66

    fig, ax  = plt.subplots(1,2)
    plt.subplots_adjust(bottom=0.25)


    for project in projects:
        project_path = os.path.join(path_head, project)
    
        C_arr = []
        eival_dict = {}
        eivec_dict = {}
        grid_dict = {}

        for path in os.listdir(project_path):

            size_path = os.path.join(project_path, path)
            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)  

            C = info["C"]
            C_arr.append(C)

        C_arr = sort(C_arr)

        
        for C in C_arr:

            size_path = os.path.join(project_path, f"C = {C}")

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
            chi_init, elas_variant, chi_noise_scale, mu_noise_scale = miscellaneous["init paras"]
        
            rng = np.random.default_rng(seed) # This is possably questionable but the functions I want to call actually does not need rng       
            sys_init = system_init(T, kappa, rng, C, mag_elas, theta, elas_variant)
            strain_cord_dict = sys_init.strain_cord_gen()
            ul_dict = sys_init.link_dict_gen()

            grid = np.empty((0,2))

            for i in range(1, int((T+1)*(T+2)/2+1)):
            
                cords = np.array([strain_cord_dict[i][1]])
                grid = np.append(grid, cords, axis = 0)

            eival_dict.update({C : eival})
            eivec_dict.update({C : eivec})
            grid_dict.update({C : grid})
        
     
    colors = []
    cmap = clr.LinearSegmentedColormap.from_list("white_red_blac", [(0, "white"), (1/9, "red"), (1, "black")], gamma=0.6)

    C_current = 0
    eigen_current = 540

    ax[0].scatter(grid_dict[C_arr[C_current]][:,0], grid_dict[C_arr[C_current]][:,1] , s = 20, c = [cmap(np.absolute(x)**2) for x in eivec_dict[C_arr[C_current]][:,eigen_current-1]], marker = "o", zorder = 3)

    for link in ul_dict:

        s, e = ul_dict[link]

        x = [grid_dict[C_arr[0]][s-1][0],grid_dict[C_arr[0]][e-1][0]]
        y = [grid_dict[C_arr[0]][s-1][1], grid_dict[C_arr[0]][e-1][1]]

        ax[0].plot(x, y, c = "grey", linewidth = 1, zorder = 2)

    ax[1].hist(eival_dict[C_arr[0]], bins)
    ax[1].axvline(eival_dict[C_arr[C_current]][eigen_current], color = "red", zorder = 3)

    ax_C_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    C_slider = Slider(ax = ax_C_slider, label = 'C', valmin = 0, valmax = len(C_arr)-1, valinit = C_current, valstep = 1)

    ax_eigen_slider = plt.axes([0.2, 0.05, 0.6, 0.05])
    eigen_slider = Slider(ax = ax_eigen_slider, label = 'E', valmin = 0, valmax = int((T+1)*(T+2)/2-1), valinit = eigen_current, valstep = 1)


    C_slider.valtext.set_text(C_arr[C_current])
 
    params = (
    rf"T = {T}" "\n"
    rf"$\kappa$ = {kappa}" "\n"
    rf"$\beta$ = {beta}" "\n"
    rf"me-coupling = {mag_elas}"
    )

    ax[0].text(0.02, 0.98, params,transform=ax[0].transAxes,va="top",ha="left",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))            

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
    
        ax[0].text(0.02, 0.98, params,transform=ax[0].transAxes,va="top",ha="left",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))            
        C_slider.valtext.set_text(C_arr[i])

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



def J_t():

    bins = 15

    fig, ax  = plt.subplots(1,3)  
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
    
        J_dict = {}
        t_dict = {}
        chi_abs_dict = {}
        C_arr = []

        for path in os.listdir(project_path):

            size_path = os.path.join(project_path, path)
            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)  

            C = info["C"]
            C_arr.append(C)

        C_arr = sort(C_arr)

        
        for C in C_arr:

            size_path = os.path.join(project_path, f"C = {C}")

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

            J_dict.update({C : []})
            chi_abs_dict.update({C : []})
            t_dict.update({C: []})

            for x in pop_link_dict.keys():
                s, e, J, chi = pop_link_dict[x]

                J_dict[C].append(J)
                chi_abs_dict[C].append(np.absolute(chi))
                t_dict[C].append(np.absolute(J*chi*(1+kappa/2 - 4*kappa*np.absolute(chi)**2)))

    ax[0].hist(J_dict[C_arr[0]], bins = bins)
    ax[1].hist(chi_abs_dict[C_arr[0]], bins = bins)
    ax[2].hist(t_dict[C_arr[0]], bins = bins)

    values = np.linspace(0, 20, 1000)
    
    C_current = 0
    ax_C_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    C_slider = Slider(
            ax = ax_C_slider,
            label = 'C',
            valmin = 0,
            valmax = len(C_arr)-1,
            valinit = C_current,
            valstep = 1
            )

    def C_update(val):
        i = int(C_slider.val)
        for axes in ax:
            axes.clear()
        ax[0].hist(J_dict[C_arr[i]], bins = bins)
        ax[1].hist(chi_abs_dict[C_arr[i]], bins = bins)
        ax[2].hist(t_dict[C_arr[i]], bins = bins)
        return
       
    C_slider.on_changed(C_update)
    plt.show()

def local_chemical_potential():

    mu_min = 0
    mu_max = 2

    cmap = clr.LinearSegmentedColormap.from_list("white_red_blac", [(0, "white"), (0.33, "yellow"), (0.66, "orange"), (1, "red")],  gamma=0.6)
    norm = clr.Normalize(vmin=mu_min, vmax=mu_max)

    fig, ax  = plt.subplots()  
    plt.subplots_adjust(bottom=0.25)

    for project in projects:
        project_path = os.path.join(path_head, project)
       
        C_arr = []
        mu_dict = {}
        strain_cord_dict_dict = {}
        grid_dict = {}

        for path in os.listdir(project_path):

            size_path = os.path.join(project_path, path)
            with open(os.path.join(size_path, "info.pkl"), "rb") as f:
                info = pickle.load(f)  

            C = info["C"]
            C_arr.append(C)

        C_arr = sort(C_arr)

        
        for C in C_arr:

            size_path = os.path.join(project_path, f"C = {C}")

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

            mu_dict.update({ C : mu_arr})

            with open(os.path.join(size_path, "miscellaneous.pkl"), "rb") as f:
                miscellaneous = pickle.load(f)
            seed = miscellaneous["seed"]
            chi_init, elas_variant, chi_noise_scale, mu_noise_scale = miscellaneous["init paras"]
        
            rng = np.random.default_rng(seed) # This is possably questionable but the functions I want to call actually does not need rng       
            sys_init = system_init(T, kappa, rng, C, mag_elas, theta, elas_variant)
            strain_cord_dict = sys_init.strain_cord_gen()

            i = 0
            if i == 0:
                ul_dict = sys_init.link_dict_gen()
                i += 1

            grid = np.empty((0,2))

            for i in range(1, int((T+1)*(T+2)/2+1)):
            
                cords = np.array([strain_cord_dict[i][1]])
                grid = np.append(grid, cords, axis = 0)

            grid_dict.update({C : grid})


    def mu_plot(i): 
        for link in ul_dict:

            s, e = ul_dict[link]

            x = [grid_dict[C_arr[i]][s-1][0],grid_dict[C_arr[i]][e-1][0]]
            y = [grid_dict[C_arr[i]][s-1][1], grid_dict[C_arr[i]][e-1][1]]

            ax.plot(x, y, c = "grey", linewidth = 1, zorder = 1)

        ax.scatter(grid_dict[C_arr[i]][:, 0], grid_dict[C_arr[i]][:, 1], color = cmap(norm(mu_dict[C_arr[i]])))


    

    C_current = 0
    mu_plot(C_current)

    sm = cm.ScalarMappable(norm = norm, cmap = cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = ax) 

    ax_C_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    C_slider = Slider(
            ax = ax_C_slider,
            label = 'C',
            valmin = 0,
            valmax = len(C_arr)-1,
            valinit = C_current,
            valstep = 1
            )

    C_slider.valtext.set_text(C_arr[C_current])

    ax.set_title(f"{project}", size = "x-large")
 
    params = (
    rf"T = {T}" "\n"
    rf"$\kappa$ = {kappa}" "\n"
    rf"$\beta$ = {beta}" "\n"
    rf"me-coupling = {mag_elas}"
    )

    ax.text(0.02, 0.98, params,transform=ax.transAxes,va="top",ha="left",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def C_update(value):
        i = int(C_slider.val)
        ax.clear()
        mu_plot(i)


        
        ax.text(0.02, 0.98, params,transform=ax.transAxes,va="top",ha="left",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        C_slider.valtext.set_text(C_arr[i])
        ax.set_title(f"{project}", size = "x-large")
        return

          
    C_slider.on_changed(C_update)
    plt.show()




if __name__ == "__main__":


    projects = {
            # 'T=45_lin_minus_pi_half_1008_01' : "-pi/2, lin",
            # 'T=45_exp_minus_pi_half_1008_01' : "-pi/2, exp",
            # 'T=45_exp_pi_half_0508_01' : "pi/2, exp",
            # 'T=45_lin_pi_half_0508_01' : "pi/2, lin",
            # 'T=47_exp_pi_half_1108_01'  : "T=47",
            # 'T=45_exp_up_1308_01' : "T=45, exp, up",
            'T=65_exp_up_1308_01' : "T=65, exp, up",            
            # 'T=65_lin_up_1708_01' : "T=65, lin, up",
            # 'T=45_exp_rot_pi_half_1708_01' : "T=45 exp, rot, pi/2" 
            # 'T=65_exp_rot_up_1708_01' : "T=65 exp, rot, up" # bad convergence!!
            }
    mode = 'pi/2' # (for real_space) options: 'flux' shows total flux, 'pi/2' shows deviations from pi/2, '-pi/2' shows derivation from -pi/2
    
    # DOS()
    # LDOS()
    # real_space(mode)
    # localization_real_space()
    # J_t()
    local_chemical_potential()


