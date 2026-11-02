import numpy as np
import matplotlib.pyplot as plt 

grid_size_x = 80
grid_size_y = 80
k_min_x = -3*np.pi/5
k_max_x = 3*np.pi/5
k_min_y = -3*np.pi/5 
k_max_y = 3*np.pi/5

kx_arr = np.linspace(k_min_x, k_max_x, grid_size_x)
ky_arr = np.linspace(k_min_y, k_max_y, grid_size_y)

KX, KY = np.meshgrid(kx_arr, ky_arr)

class NN:

    def __init__(self, kx, ky):

        self.kx = kx
        self.ky = ky

    def q1(self):

        return self.kx

    def q2(self):

        return (self.kx + np.sqrt(3)*self.ky)/2

    def q3(self):

        return(self.kx - np.sqrt(3)*self.ky)/2

def Ham_builder(kx, ky):

    NNv = NN(kx,ky)

    Ham = np.array([
        [0, np.exp(1j*NNv.q2()), -np.exp(-1j*NNv.q3()), 0, np.exp(-1j*NNv.q2()), np.exp(1j*NNv.q3())],
        [np.exp(-1j*NNv.q2()), 0, 2*1j*np.sin(NNv.q1()), np.exp(1j*NNv.q2()), 0, 0],
        [-np.exp(1j*NNv.q3()), -2*1j*np.sin(NNv.q1()), 0, np.exp(-1j*NNv.q3()), 0, 0],
        [0, np.exp(-1j*NNv.q2()), np.exp(1j*NNv.q3()), 0, np.exp(1j*NNv.q2()), np.exp(-1j*NNv.q3())],
        [np.exp(1j*NNv.q2()), 0 ,0, np.exp(-1j*NNv.q2()), 0, 2*np.cos(NNv.q1())],
        [np.exp(-1j*NNv.q3()), 0, 0, np.exp(1j*NNv.q3()), 2*np.cos(NNv.q1()), 0]
        ])

    return Ham

bands = np.zeros((6,grid_size_y, grid_size_x))

for i in range(0 , grid_size_y):

    for j in range(0, grid_size_x):

        eival, eivec = np.linalg.eigh(Ham_builder(KX[i][j], KY[i][j]))

        for l in [0,1,2,3,4,5]:
        
            bands[l][i][j] = eival[l]




fig = plt.figure()

def wireframe():

    fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

    for i in [0,1,2,3,4,5]:
        ax.plot_wireframe(KX, KY, bands[i], color = colors[i])

    
    # ax.set_xticks([-np.pi/2, np.pi/2])
    # ax.set_xticklabels([r"$-\frac{\pi}{2}$", r"$\frac{\pi}{2}$"])
    # ax.xaxis.set_tick_params(pad = 0)
    # ax.set_yticks([-np.pi/2, np.pi/2])
    # ax.set_yticklabels([r"$-\frac{\pi}{2}$", r"$\frac{\pi}{2}$"])
    # ax.yaxis.set_tick_params(pad = 0)
    

    ax.set_xticks([])
    ax.set_yticks([])

    ax.view_init(elev = 10, azim = 65, vertical_axis = 'z')

    ax.set_xlabel(f"$p_x$", size = "x-large")
    ax.set_ylabel(f"$p_y$", size = "x-large")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("E", rotation = "horizontal" ,  size = "x-large")

    
    plt.show()
    fig.savefig("/home/erick/Documents/Uni/Master-Project/wiki/WARB_presentation/pictures/bands_kagome.png", bbox_inches="tight", pad_inches=0.02, dpi=1200)



def dispersion():

    ax = fig.add_subplot(111)
    c = ax.pcolormesh(KX, KY, bands[3])
    colorbar = fig.colorbar(c, ax=ax)

    levels = np.linspace(-1, 1, 11)
    contours = plt.contour(KX, KY, bands[3], levels = levels, colors = "black", linestyle = "solid", linewidth = 0.5)

    ax.plot([np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2], [-np.pi/(2*np.sqrt(3)), np.pi/(2*np.sqrt(3)), np.pi/(2*np.sqrt(3)),    -np.pi/(2*np.sqrt(3)), -np.pi/(2*np.sqrt(3))], c = "black", ls= '--', lw = 2)

    # to band below:
    # ax.text(np.pi/4, np.pi/(4*np.sqrt(3)), r"$\vec{A}$", ha = "center", va = "center")
    # ax.text(-np.pi/4, -np.pi/(4*np.sqrt(3)), r"$-\vec{A}$", ha = "center", va = "center")
    # ax.scatter(np.pi/4, np.pi/(4*np.sqrt(3)), s = 50 , marker = "x" , c = "black")
    # ax.scatter(-np.pi/4, -np.pi/(4*np.sqrt(3)), s = 50 , marker = "x" , c = "black")
    # ax.text(np.pi/4 + 0.15, np.pi/(4*np.sqrt(3)) + 0.05, r"$F$", fontsize = "x-large")
    # ax.text(-np.pi/4 + 0.15, -np.pi/(4*np.sqrt(3)) + 0.05, r"$F^{\prime}$", fontsize = "x-large")
    
    fontsize = "small"

    # to band above: 
    ax.text(np.pi/12, -np.pi/(4*np.sqrt(3)), r"$\vec{\text{A}}$", ha = "center", va = "center", fontsize = fontsize)
    ax.text(-np.pi/12, np.pi/(4*np.sqrt(3)), r"$-\vec{\text{A}}$", ha = "center", va = "center", fontsize = fontsize)
    # ax.text(np.pi/12 + 0.15, -np.pi/(4*np.sqrt(3)) + 0.05 , r"C", fontsize = "x-large")
    # ax.text(-np.pi/12 + 0.15, np.pi/(4*np.sqrt(3)) + 0.05 , r"$C^{\prime}$", fontsize = "x-large")
    

    ax.text(5*np.pi/12, -np.pi/(4*np.sqrt(3)), r"$-\vec{\text{A}}$", ha = "center", va = "center", fontsize = fontsize)
    ax.text(-5*np.pi/12, np.pi/(4*np.sqrt(3)), r"$\vec{\text{A}}$", ha = "center", va = "center", fontsize = fontsize)
    # ax.text(5*np.pi/12 - 0.325, -np.pi/(4*np.sqrt(3)) + 0.05, r"E", fontsize = "x-large")
    # ax.text(-5*np.pi/12 + 0.15, np.pi/(4*np.sqrt(3)) + 0.05, r"$E^{\prime}$", fontsize = "x-large")

    # ax.tick_params( axis = 'both', which = 'both', bottom = False, left = False, labelbottom = False, labelleft = False)

    ax.set_xticks([-np.pi/2, 0, np.pi/2])
    ax.set_xticklabels([r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$"], fontsize = "x-large")
    ax.set_yticks([-np.pi/2, 0, np.pi/2])
    ax.set_yticklabels([r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$"], fontsize = "x-large")
    ax.set_xlabel(f"$p_x$", fontsize = "x-large")
    ax.set_ylabel(f"$p_y$", fontsize = "x-large")

    colorbar.set_label("E", rotation = 0, labelpad = 10, size = "x-large")
    plt.show()

    fig.savefig("/home/erick/Documents/Uni/Master-Project/wiki/WARB_presentation/pictures/dispersion_kagome.png", bbox_inches="tight", pad_inches=0.02, dpi=300)    # ax.grid()


# 4th band:

# ax.set_xticks(np.pi*np.array([-0.25, 0, 0.25]))
# ax.set_xticklabels([r'$-\frac{\pi}{4}$', r'0', r'$\frac{\pi}{4}$'], size='xx-large')

# ax.set_yticks(np.pi*np.array([-0.25, 0, 0.25])/np.sqrt(3))
# ax.set_yticklabels([r'$-\frac{\pi}{4\sqrt{3}}$', '0', r'$\frac{\pi}{4\sqrt{3}}$'], size = 'xx-large')

# 5th band:

# ax.set_xticks(np.pi*np.array([-1, -0.5, 0, 0.5, 1]))

# ax.set_xticks(np.pi*np.array([-0.125, 0, 0.125])/np.sqrt(3))
# ax.set_xticklabels([r'$-\frac{\pi}{8\sqrt{3}}$', '0', r'$\frac{\pi}{8\sqrt{3}}$']) 

# ax.set_yticks(np.pi*np.array([-0.25, 0, 0.25])/np.sqrt(3))
# ax.set_yticklabels([r'$-\frac{\pi}{4\sqrt{3}}$', '0', r'$\frac{\pi}{4\sqrt{3}}$'], size = 'xx-large')

# ax.set_xticks([0, -np.pi/12])
# ax.set_yticks([0, np.pi/(4*np.sqrt(3))])

# ind = np.unravel_index(np.argmin(bands[4]), np.shape(bands[4]))



# print(np.array([KX[ind], 4*np.sqrt(3)*KY[ind]/np.pi]))
# print(np.array([KX[ind], KY[ind]]))

dispersion()
