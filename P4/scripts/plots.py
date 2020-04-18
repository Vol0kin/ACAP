import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

headers = ['Tamaño vector', 'Tiempo (s)']
df_seq = pd.read_csv('../out/sequential.csv', names=headers)
df_cuda = pd.read_csv('../out/cuda.csv', names=headers)

seq_times = df_seq['Tiempo (s)'].values
cuda_times = df_cuda['Tiempo (s)'].values
speedup = seq_times / cuda_times
print(speedup)

"""
df_seq.plot(kind='bar', x='Tamaño vector', y='Tiempo (s)', color='red', rot=0)
plt.grid()
plt.show()

df_cuda.plot(kind='bar', x='Tamaño vector', y='Tiempo (s)', color='blue', rot=0)
plt.grid()
plt.show()
"""

# Plot times
big_df = pd.DataFrame({'Secuencial': seq_times, 'CUDA': cuda_times}, index=df_seq['Tamaño vector'].values)
big_df.plot(kind='bar', rot=0)

plt.xlabel('Tamaño vector')
plt.ylabel('Tiempo (s)')
plt.yscale('log')

plt.grid()
plt.show()

# Plot speedup
ax = plt.gca()
x = [n for n in range(len(speedup))]

ax.set_xticks(x)
ax.set_xticklabels(df_seq['Tamaño vector'].values)

plt.plot(x, speedup, 'g-o')
plt.xlabel('Tamaño vector')
plt.ylabel('Ganancia')

plt.grid()
plt.show()
