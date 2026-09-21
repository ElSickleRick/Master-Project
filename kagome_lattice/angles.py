import numpy as np

phi = np.arccos(-(1+np.sqrt(3))/4)
kappa = np.arccos(-(1-np.sqrt(3))/np.sqrt(12-2*np.sqrt(3)))
theta = np.arccos(-(1-np.sqrt(3))/np.sqrt(12-2*np.sqrt(3)))
# kappa = np.arccos((1-np.sqrt(3))/(1+np.sqrt(3))*(np.cos(phi)/np.sin(phi))
# theta = np.arccos((1-np.sqrt(3))/(4*np.sin(phi)))

sigmazpx = ((1-np.sqrt(3))/2)*np.cos(phi)-((1+np.sqrt(3))/2)*np.sin(phi)*np.cos(kappa)
sigmazpy = -((1+np.sqrt(3))/2)*np.cos(phi)-((1-np.sqrt(3))/2)*np.sin(phi)*np.cos(kappa)

sigmaxpx = ((1-np.sqrt(3))/2)*np.sin(phi)*np.cos(theta)-((1+np.sqrt(3))/2)*(np.sin(theta)*np.sin(kappa) - np.cos(phi)*np.cos(theta)*np.cos(kappa))
sigmaxpy = -((1+np.sqrt(3))/2)*np.sin(phi)*np.cos(theta)-((1-np.sqrt(3))/2)*(np.sin(theta)*np.sin(kappa) - np.cos(phi)*np.cos(theta)*np.cos(kappa))

sigmaypx = -((1-np.sqrt(3))/2)*np.sin(phi)*np.sin(theta) - ((1+np.sqrt(3))/2)*(np.cos(theta)*np.sin(kappa)+np.cos(phi)*np.sin(theta)*np.cos(kappa))
sigmaypy = ((1+np.sqrt(3))/2)*np.sin(phi)*np.sin(theta)-((1-np.sqrt(3))/2)*(np.cos(theta)*np.sin(kappa)+np.cos(phi)*np.sin(theta)*np.sin(kappa))


#print(sigmazpx)
#print(sigmazpy)
#print(sigmaxpx)
#print(sigmaxpy)
#print(sigmaypx)
#print(sigmaypy)

print(phi)
print(kappa)
print(theta)
