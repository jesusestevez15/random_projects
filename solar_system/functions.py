import numpy as np

def Aceleracion(m, x, N, G):
    a = np.zeros((N,2))
    for i in range(N):
        for j in range(N):
            if j != i:
                GMX = -G*m[j]/((x[i][0] - x[j][0])**2 + (x[i][1] - x[j][1])**2)**1.5
                a[i][0] = a[i][0] + GMX*(x[i][0] - x[j][0])
                a[i][1] = a[i][1] + GMX*(x[i][1] - x[j][1])
    return a

def Posicion(x, v, w, h, a, N):
    for i in range(N):
        x[i][0] = x[i][0] + h*v[i][0] + a[i][0]*h**2/2
        x[i][1] = x[i][1] + h*v[i][1] + a[i][1]*h**2/2
        w[i][0] = v[i][0] + a[i][0]*h/2
        w[i][1] = v[i][1] + a[i][1]*h/2
    return x, w

def Velocidad(v, w, a, h, N):
    for i in range(N):
        v[i][0] = w[i][0] + a[i][0]*h/2
        v[i][1] = w[i][1] + a[i][1]*h/2
    return v

def Impacto(m, x, v, w, a, N, threshold):
    i = 0
    while i < N:
        j = 0
        while j < N:
            if j != i:
                diff = ((x[i][0] - x[j][0])**2 + (x[i][1] - x[j][1])**2)**0.5
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