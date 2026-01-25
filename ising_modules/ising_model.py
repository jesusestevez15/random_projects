import numpy as np
import matplotlib.pyplot as plt
import random

N = 100  # tamaño de la red de los espines
T = 5 # temperatura
s = np.ones((N + 2,N + 2))  # configuración inicial de los espines S

# Método Monte Carlo
for num in range(1, 10):
    for pMC in range(1, N*N):

        # Energía de la configuración de espines S

        #suma = 0

        #for i in range(1, N + 1):
        #    for j in range(1, N + 1):
        #        suma = suma + s[i, j]*  (s[i, j + 1] + s[i, j - 1] + s[i + 1, j] + s[i - 1, j])

        #E = -0.5 * suma

        # Cojo un punto aleatorio

        n = random.randint(1,N)
        m = random.randint(1,N)

        # Condiciones de contorno

        s[0, m] = s[N, m]
        s[N + 1, m] = s[1, m]
        s[n, 0] = s[n, N]
        s[n, N + 1] = s[n, 1]

        # Evalúo p (probabilidad de cambiar de estado)

        dE = 2 * s[n, m] * (s[n + 1, m] + s[n - 1, m] + s[n, m + 1] + s[n, m - 1]) 
        p = min(1, np.exp(-dE/T))

        # Compruebo si se efectúa el cambio de estado

        eps = random.uniform(0, 1)

        if eps < p:
            s[n, m] = -s[n, m]

for i in range(1, N + 1):
    for j in range(1, N + 1):
        if s[i, j] == 1:
            plt.plot(i, j, '.', color='blue')