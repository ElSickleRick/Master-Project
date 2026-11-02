import numpy as np

eta = 1/2
gamma = 1/2


def bloch_vector(cords):

    qx, qy = cords

    hx = 2*np.cos((qx+np.sqrt(3)*qy)/2)*(1-eta+gamma)
    hy = 2*(np.sin((qx-np.sqrt(3)*qy)/2) + (eta-gamma)*np.sin((qx-np.sqrt(3)*qy)/2))   
    hz = 2*np.cos(qx)

    return [hx, hy, hz]
    

def circ_path(d_point, param):

