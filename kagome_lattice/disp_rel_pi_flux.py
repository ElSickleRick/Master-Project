import numpy as np
import matplotlib.pyplot as plt 

grid_size = 100 # in one direction
k_min = -np.pi
k_max = np.pi

kx_arr = np.linspace(k_min, k_max, grid_size)
ky_arr = np.linspace(k_min, k_max, grid_size)

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

bands = np.zeros((6,grid_size, grid_size))

for i in range(0 , grid_size):

    for j in range(0, grid_size):

        eival, eivec = np.linalg.eigh(Ham_builder(KX[i][j], KY[i][j]))

        for l in [0,1,2,3,4,5]:
        
            bands[l][i][j] = eival[l]




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

for i in [0, 1, 2, 3, 4, 5]:
    ax.plot_wireframe(KX, KY, bands[i], color = colors[i])

plt.show()


