import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


ini1 = np.array([0.010941, 0.009968, 0.010388])
ini2 = np.array([0.012234, 0.011471, 0.011520])
ini3 = np.array([0.011469, 0.012901, 0.011492])
ini4 = np.array([0.014823, 0.047699, 0.023243])

comp1 = np.array([4.130078, 4.128992, 4.129614])
comp2 = np.array([2.065498, 2.065162, 2.067519])
comp3 = np.array([1.418354, 1.418547, 1.418820])
comp4 = np.array([1.096716, 1.146678, 1.097349])

reduce1 = np.array([0.000004, 0.000006, 0.000007])
reduce2 = np.array([0.000107, 0.000122, 0.000133])
reduce3 = np.array([0.000122, 0.000134, 0.000026])
reduce4 = np.array([0.000287, 0.000030, 0.000027])

x_axis = np.arange(1, 5)

comp_times = np.array([np.mean(comp1), np.mean(comp2), np.mean(comp3), np.mean(comp4)])
ini_times = np.array([np.mean(ini1), np.mean(ini2), np.mean(ini3), np.mean(ini4)])
reduce_times = np.array([np.mean(reduce1), np.mean(reduce2), np.mean(reduce3), np.mean(reduce4)])

plt.plot(x_axis, comp_times, "bo-")
plt.ticklabel_format(style='plain',axis='x')
plt.grid()
plt.show()
