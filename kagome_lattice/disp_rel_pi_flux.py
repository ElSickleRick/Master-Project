import numpy as np
import matplotlib.pyplot as plt 

grid_size_x = 250
grid_size_y = 250
k_min_x = -np.pi/2
k_max_x = 0
k_min_y = 0
k_max_y = np.pi/2

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



ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

for i in [3,4]:
    ax.plot_wireframe(KX, KY, bands[i], color = colors[i])

"""

ax = fig.add_subplot(111)
c = ax.pcolormesh(KX, KY, bands[3])
fig.colorbar(c, ax=ax)

ax.grid()


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

ax.set_xticks([0, -np.pi/12])
ax.set_yticks([0, np.pi/(4*np.sqrt(3))])

ind = np.unravel_index(np.argmin(bands[4]), np.shape(bands[4]))

"""

# print(np.array([KX[ind], 4*np.sqrt(3)*KY[ind]/np.pi]))
# print(np.array([KX[ind], KY[ind]]))

plt.show()
