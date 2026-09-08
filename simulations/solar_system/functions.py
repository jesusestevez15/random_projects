import numpy as np

def Aceleracion(m, x, N, G):
    # diff[i,j] = x[i] - x[j], forma (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]

    # dist2[i,j] = distancia al cuadrado entre cuerpo i y j, forma (N, N)
    dist2 = np.sum(diff**2, axis=2)

    # Evitar división por cero en la diagonal (i == j)
    np.fill_diagonal(dist2, np.inf)

    # factor[i,j] = -G*m[j] / dist^3, forma (N, N)
    factor = -G * m[np.newaxis, :] / dist2**1.5

    # a[i] = suma_j factor[i,j] * diff[i,j], forma (N, 3)
    a = np.sum(factor[:, :, np.newaxis] * diff, axis=1)

    return a

def Posicion(x, v, w, h, a, N):
    x = x + h*v + a*h**2/2
    w = v + a*h/2
    return x, w

def Velocidad(v, w, a, h, N):
    v = w + a*h/2
    return v

def Impacto(m, x, v, w, a, N, threshold):
    i = 0
    while i < N:
        j = 0
        while j < N:
            if j != i:
                diff = ((x[i][0]-x[j][0])**2 + (x[i][1]-x[j][1])**2 + (x[i][2]-x[j][2])**2)**0.5
                if diff < threshold:
                    x = np.delete(x, j, 0)
                    v[i] = (v[i] * m[i] + v[j] * m[j]) / (m[i] + m[j])
                    v = np.delete(v, j, 0)
                    w[i] = (w[i] * m[i] + w[j] * m[j]) / (m[i] + m[j])
                    w = np.delete(w, j, 0)
                    a[i] = (a[i] * m[i] + a[j] * m[j]) / (m[i] + m[j])
                    a = np.delete(a, j, 0)
                    m[i] = m[i] + m[j]
                    m = np.delete(m, j, 0)
                    N -= 1
            j += 1
        i += 1
    return m, x, v, w, a, N