import numpy as np
import matplotlib.pyplot as plt

def plot_lines(x, y1, y2, label1, label2, xlabel, ylabel, scale=False):
    plt.clf()

    plt.plot(x, y1, 'ro-', label=label1)
    plt.plot(x, y2, 'bo-', label=label2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    if scale:
        plt.yscale('log')

    plt.show()



n_proces = ['1', '2', '4']

local_small = np.array([0.733139, 0.49165, 0.394203])
local_big = np.array([26.8771, 15.6187, 9.20751])
atcgrid_small = np.array([1.10137, 0.595069, 0.326284])
atcgrid_big = np.array([52.6512, 28.9049, 15.6996])

speedup_local_small = local_small[0] / local_small
speedup_local_big = local_big[0] / local_big
speedup_atcgrid_small = atcgrid_small[0] / atcgrid_small
speedup_atcgrid_big = atcgrid_big[0] / atcgrid_big

print(f'Local small: {speedup_local_small}')
print(f'Local big: {speedup_local_big}')
print(f'atcgrid small: {speedup_atcgrid_small}')
print(f'atcgrid big: {speedup_atcgrid_big}')

plot_lines(n_proces, local_small, local_big, 'Problema pequeño', 'Problema grande', 'Núm. procesos', 'Tiempo total ejecución (s)', scale=True)
plot_lines(n_proces, atcgrid_small, atcgrid_big, 'Problema pequeño', 'Problema grande', 'Núm. procesos', 'Tiempo total ejecución (s)', scale=True)

plot_lines(n_proces, speedup_local_small, speedup_local_big, 'Problema pequeño', 'Problema grande', 'Núm. procesos', 'Ganancia')
plot_lines(n_proces, speedup_atcgrid_small, speedup_atcgrid_big, 'Problema pequeño', 'Problema grande', 'Núm. procesos', 'Ganancia')
