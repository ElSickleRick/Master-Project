import numpy as np
import matplotlib.pyplot as plt 

class analysis:

    def __init__(self, pop_link_dict):

        self.pop_link_dict = pop_link_dict 

        return

    def Chi_abs_dist_plot(self):

        Chi_abs_arr = []

        for x in self.pop_link_dict:

            Chi_abs_arr.append(np.absolute(self.pop_link_dict[x][3]))

        mean = np.mean(Chi_abs_arr)
        std = np.std(Chi_abs_arr) 

        fig, ax = plt.subplots()
        ax.hist(Chi_abs_arr, bins = 30)

        return mean, std, Chi_abs_arr 



