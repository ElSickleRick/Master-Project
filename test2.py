import numpy as np
# import numba as nb
import time 
import scipy.linalg as la 
import matplotlib.pyplot as plt 

from system_init import system_init 
from MF_loop import MF_loop
from analysis import analysis


J  = 1
k = 10
chi = 0.577
flux = np.pi

t = -J/2*(1+k/2-k*chi**2)*chi

print(t)
