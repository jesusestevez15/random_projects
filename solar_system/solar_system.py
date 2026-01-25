import functions
import numpy as np
import matplotlib.pyplot as plt
# Variables

h = 6.277698332/100 # Paso h=0.01719917351 equivale a un día terrestre
t = 0
G = 6.67*10**-11
msol = 1.9891*10**30
ua = 1.496*10**11
Tmax = 100000
N=200

# Masa del sol: 1.9891*10**30 kg
#m = np.array([1,0.000000166,0.00000244784,0.00000300346,0.00000003694,0.0000003227,0.00095455231,0.00028581267,0.00004366246,0.00005150067]) # en masas solares
#x = np.array([[0,0],[0.387099,0],[0.723336,0],[1.000003,0],[1.002570,0],[1.523710,0],[5.202887,0],[9.536676,0],[19.189165,0],[30.069923,0]]) # en UA
#v = np.array([[0,0],[0,1.607538617],[0,1.175648729],[0,1],[0,1.03423896218],[0,0.808251368],[0,0.4387424888],[0,0.3246970358],[0,0.2286078754],[0,0.1838866696]])

# Masa de planetas
m = np.random.rand(N) * 10

# Planetas random con un 99% menos de masa
m[np.random.choice(N, size=int(0.3 * N), replace=False)] *= 0.01

x = np.random.rand(N, 2) - 0.5
#v  = np.random.rand(N, 2) - 0.5

# Velocidad inicial rotacional dada
v1 = -0.1 * x[:,1]
v2 = 0.1 * x[:,0]
v = np.stack((v1,v2), axis=1)

# Estrellas binarias
m[0] = 10000000
m[8] = 10000000

# Estrella 1 estática
x[0] = 0
v[0] = 0
th = 0.02

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)

sc1 = ax1.scatter([], [])
sc11 = ax1.scatter([], [])
sc12 = ax1.scatter([], [])
sc2 = ax2.scatter([], [])
sc21 = ax1.scatter([], [])
sc22 = ax1.scatter([], [])
ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])
ax2.set_xlim([-0.1,0.1])
ax2.set_ylim([-0.1,0.1])

# Crear texto dinámico (posición y contenido inicial)
text_info1 = ax1.text(0.4, 0.95, '', transform=ax1.transAxes, fontsize=10)
text_info2 = ax2.text(0.4, 0.95, '', transform=ax2.transAxes, fontsize=10)

a = np.zeros((N,2))
w = np.zeros((N,2))
a = functions.Aceleracion(m, x, N, G)

while t < Tmax:
    x, w = functions.Posicion(x, v, w, h, a, N)
    a = functions.Aceleracion(m, x, N, G)
    v = functions.Velocidad(v, w, a, h, N)
    m, x, v, w, a, N = functions.Impacto(m, x, v, w, a, N, th)

    t = t + h
    
    sc1.remove()
    sc11.remove()
    sc12.remove()
    sc1 = ax1.scatter(x[:, 0], x[:, 1], color='blue', marker='.', linewidths=2)
    sc11 = ax1.scatter(x[0, 0], x[0, 1], color='green', marker='.', linewidths=2)
    sc12 = ax1.scatter(x[8, 0], x[8, 1], color='green', marker='.', linewidths=2)
    text_info1.set_text(f'Tiempo: {t:.2f}  |  Número de cuerpos: {N}')
    sc2.remove()
    sc21.remove()
    sc22.remove()
    sc2 = ax2.scatter(v[:, 0], v[:, 1], color='red', marker='.', linewidths=2)
    sc21 = ax2.scatter(v[0, 0], v[0, 1], color='green', marker='.', linewidths=2)
    sc22 = ax2.scatter(v[8, 0], v[8, 1], color='green', marker='.', linewidths=2)
    text_info2.set_text(f'Tiempo: {t:.2f}  |  Número de cuerpos: {N}')   
    plt.draw()
    plt.pause(0.01)
plt.ioff()
plt.show()
