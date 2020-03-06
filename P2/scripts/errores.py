import numpy as np
import matplotlib.pyplot as plt

def plot_lines(x_axis, y_axis, color, title, x_label, y_label, scale=False):
    # Limpiar estilo
    plt.clf()
    
    plt.plot(x_axis, y_axis, color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if scale:
        plt.yscale('log')
    
    plt.grid()
    
    plt.show()


# Errores obtenidos
error_pm = 0.000000000000062172489379
error_def = 0.000000100000193903326817 # En el punto izquierdo
error_exc = 0.000000099999806213446618 # En el punto derecho

# Dibujar grafico de barras de los errores
width = 0.4
x_ticks = np.arange(3)

plt.bar(x_ticks, (error_def, error_pm, error_exc), width)
plt.xticks(x_ticks, ('Defecto', 'Punto medio', 'Exceso'))
plt.title('Error cometido según el punto tomado')
plt.ylabel('Error')
plt.xlabel('Tipo de error')
plt.yscale('log')
plt.show()


evolucion_errores = np.array([0.000000000008368417070415, 0.000000000000028865798640,
                              0.000000000000062172489379, 0.000000000000633271213246,
                              0.000000000000177635683940, 0.000000000000045741188615])

intervalos = ('100000', '1000000', '10000000', '100000000', '1000000000', '2000000000')
tiempos = (0.000488, 0.004982, 0.047928, 0.476627, 4.650241, 9.254434)

plot_lines(intervalos, evolucion_errores, 'ro-', 'Evolución del error según el número de intervalos', 'Num. intervalos', 'Error', scale=True)
plot_lines(intervalos, tiempos, 'bo-', 'Evolución del tiempo de ejecución según el número de intervalos', 'Num. intervalos', 'Tiempo (s)')
