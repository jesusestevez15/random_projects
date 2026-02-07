"""
Este módulo contiene funciones para calcular números primos
y guardarlos en un archivo de texto.
Fecha: 2026-02-07
"""

import time
import numpy as np

print('Introduce un numero:')
numero = int(input())

inicio = time.time()
total = int(np.floor(np.sqrt(numero)))
x = np.ones(numero-1)
x[:2] = 0

for i in range(2,total+1):
    j = 2
    while i * j < numero:
        x[i * j - 1] = 0
        j = j + 1

v = [i+1 for i in range(numero-1) if x[i] != 0]

final =time.time()

v = np.round(v).astype(int)
print("Tiempo de ejecucion: ", final-inicio, " segundos")

with open("prime_numbers/numbers.txt", "w") as f:
    f.write(f"Los numeros primos del 1 al {numero} son:\n")
    np.savetxt(f, v, fmt="%d")
