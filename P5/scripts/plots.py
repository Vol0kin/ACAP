import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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



n_pixels = ['1', '2', '4']

small = np.array([0.708568, 0.296242, 0.250583])
big = np.array([44.9549, 17.9155, 15.6332])

speedup_small = 0.520567 / small
speedup_big = 27.5843 / big


print(f'Small: {speedup_small}')
print(f'Big: {speedup_big}')

plot_lines(n_pixels, small, big, 'Problema pequeño', 'Problema grande', 'Núm. píxeles calculados por hebra por cada dimensión', 'Tiempo total ejecución (s)', scale=True)
plot_lines(n_pixels, speedup_small, speedup_big, 'Problema pequeño', 'Problema grande', 'Núm. píxeles calculados por hebra por cada dimensión', 'Ganancia')


plt.clf()

index = ['Tamaño pequeño', 'Tamaño grande']
seq = [0.520567, 27.5843]
mpi = [0.316762, 10.0926]
cuda = [0.250583, 15.6332]

df = pd.DataFrame(np.c_[seq, mpi, cuda], index=index, columns=["Secuencial", "MPI", "CUDA"])
df.plot.bar(rot=0)

plt.xlabel('Tamaño del problema')
plt.ylabel('Tiempo (s)')
plt.yscale('log')

plt.show()
