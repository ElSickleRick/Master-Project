import numpy as np

phi = -np.pi/2
kappa = -np.pi/2
theta = 5*np.pi/12
# kappa = np.arccos((1-np.sqrt(3))/(1+np.sqrt(3))*(np.cos(phi)/np.sin(phi))
# theta = np.arccos((1-np.sqrt(3))/(4*np.sin(phi)))

sigmazpx = ((1-np.sqrt(3))*np.cos(phi)-(1+np.sqrt(3))*np.sin(phi)*np.cos(kappa))/np.sqrt(8)
sigmazpy = (-(1+np.sqrt(3))*np.cos(phi)-(1-np.sqrt(3))*np.sin(phi)*np.cos(kappa))/np.sqrt(8)

sigmaxpx = ((1-np.sqrt(3))*np.sin(phi)*np.cos(theta)-(1+np.sqrt(3))*(np.sin(kappa)*np.sin(theta) - np.cos(phi)*np.cos(kappa)*np.cos(theta)))/np.sqrt(8)
sigmaxpy = (-(1+np.sqrt(3))*np.sin(phi)*np.cos(theta)-(1-np.sqrt(3))*(np.sin(kappa)*np.sin(theta) - np.cos(phi)*np.cos(kappa)*np.cos(theta)))/np.sqrt(8)

sigmaypx = (-(1-np.sqrt(3))*np.sin(phi)*np.sin(theta) - (1+np.sqrt(3))*(np.sin(kappa)*np.cos(theta)+np.cos(phi)*np.cos(kappa)*np.sin(theta)))/np.sqrt(8)
sigmaypy = ((1+np.sqrt(3))*np.sin(phi)*np.sin(theta)-(1-np.sqrt(3))*(np.sin(kappa)*np.cos(theta)+np.cos(phi)*np.cos(kappa)*np.sin(theta)))/np.sqrt(8)


print(sigmazpx)
print(sigmazpy)
print(sigmaxpx)
print(sigmaxpy)
print(sigmaypx)
print(sigmaypy)

# print(phi)
# print(kappa)
# print(theta)
