import functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necesario para projection='3d'

# Variables
h = 6.277698332/100
t = 0
G = 6.67*10**-11
msol = 1.9891*10**30
ua = 1.496*10**11
Tmax = 100000
N = 200

# Masa de planetas
m = np.random.rand(N) * 10
m[np.random.choice(N, size=int(0.3 * N), replace=False)] *= 0.01

# Posiciones iniciales en 3D
x = np.random.rand(N, 3) - 0.5

# Velocidad inicial rotacional usando producto vectorial (disco tipo protoplanetario en plano XY)
eje_rotacion = np.array([0, 0, 1])
omega = 0.3
v = omega * np.cross(eje_rotacion, x)

# Estrellas binarias
m[0] = 10000000
m[8] = 10000000

# Estrella 1 estática
x[0] = 0
v[0] = 0
th = 0.02

plt.ion()
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

a = np.zeros((N,3))
a = functions.Aceleracion(m, x, N, G)

while t < Tmax:
    x, w = functions.Posicion(x, v, np.zeros((N,3)), h, a, N)
    a = functions.Aceleracion(m, x, N, G)
    v = functions.Velocidad(v, w, a, h, N)
    m, x, v, w, a, N = functions.Impacto(m, x, v, w, a, N, th)

    t = t + h

    ax1.clear()
    ax1.set_xlim([-1,1]); ax1.set_ylim([-1,1]); ax1.set_zlim([-1,1])
    ax1.scatter(x[:,0], x[:,1], x[:,2], color='blue', marker='.')
    ax1.scatter(x[0,0], x[0,1], x[0,2], color='green', marker='.')
    if N > 8:
        ax1.scatter(x[8,0], x[8,1], x[8,2], color='green', marker='.')
    ax1.set_title(f'Tiempo: {t:.2f}  |  Número de cuerpos: {N}')

    ax2.clear()
    ax2.set_xlim([-0.1,0.1]); ax2.set_ylim([-0.1,0.1]); ax2.set_zlim([-0.1,0.1])
    ax2.scatter(v[:,0], v[:,1], v[:,2], color='red', marker='.')
    ax2.scatter(v[0,0], v[0,1], v[0,2], color='green', marker='.')
    if N > 8:
        ax2.scatter(v[8,0], v[8,1], v[8,2], color='green', marker='.')
    ax2.set_title(f'Tiempo: {t:.2f}  |  Número de cuerpos: {N}')

    plt.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()