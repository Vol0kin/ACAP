import numpy as np
import matplotlib.pyplot as plt

def plot_lines(x_axis, y_axis, color, title, x_label, y_label):
    # Limpiar estilo
    plt.clf()
    
    plt.plot(x_axis, y_axis, color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.grid()
    
    plt.show()
    

# Crear arrays con los tiempos obtenidos
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

x_axis = ['1', '2', '3', '4']

# Calcular valores medios
ini_medio = np.array([np.mean(ini1), np.mean(ini2), np.mean(ini3), np.mean(ini4)])
comp_medio = np.array([np.mean(comp1), np.mean(comp2), np.mean(comp3), np.mean(comp4)])
reduce_medio = np.array([np.mean(reduce1), np.mean(reduce2), np.mean(reduce3), np.mean(reduce4)])

# Calcular desviaciones
ini_std = np.array([np.std(ini1), np.std(ini2), np.std(ini3), np.std(ini4)])
comp_std = np.array([np.std(comp1), np.std(comp2), np.std(comp3), np.std(comp4)])
reduce_std = np.array([np.std(reduce1), np.std(reduce2), np.std(reduce3), np.std(reduce4)])

# Calcular ganancias
ganancia = comp_medio[0] / comp_medio

# Imprimir informacion
print(f'Tiempo medio de inicializacion: {ini_medio}, std: {ini_std}')
print(f'Tiempo medio de computo: {comp_medio}, std: {comp_std}')
print(f'Tiempo medio de reduccion: {reduce_medio}, std: {reduce_std}')
print(f'Ganancia: {ganancia}')

# Dibujar valores medios de inicializacion
plot_lines(x_axis, ini_medio, 'ro-', 'Evolución del tiempo de inicialización', 'Num. procesos', 'Tiempo (s)')

# Dibujar valores medios de computo
plot_lines(x_axis, comp_medio, 'bo-', 'Evolución del tiempo de cómputo', 'Num. procesos', 'Tiempo (s)')

# Dibujar valores medios de reduccion
plot_lines(x_axis, reduce_medio, 'go-', 'Evolución del tiempo de recepción', 'Num. procesos', 'Tiempo (s)')

width = 0.4
x_ticks = np.arange(4)

plt.clf()
#plt.grid()

p1 = plt.bar(x_ticks, ini_medio, width)
p2 = plt.bar(x_ticks, comp_medio, width, bottom=ini_medio)
p3 = plt.bar(x_ticks, reduce_medio, width, bottom=ini_medio + comp_medio)

plt.xticks(x_ticks, x_axis)
plt.yticks(np.arange(0, 5, 0.5))
plt.legend((p1[0], p2[0], p3[0]), ('Inicializacion', 'Cómputo', 'Recepción'))
plt.title('Tiempo total de ejecución según número de procesos')
plt.xlabel('Tiempo (s)')
plt.ylabel('Num. procesos')
plt.show()

# Dibujar ganancia
plot_lines(x_axis, ganancia, 'ko-', 'Evolución de la ganancia', 'Num. procesos', 'Ganancia en velocidad')
