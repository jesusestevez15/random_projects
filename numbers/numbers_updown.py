import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Función que genera la secuencia de Collatz
def collatz_sequence(n):
    v = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        v.append(n)
    return v

# Valor inicial
numero_inicial = 10
min_num = 1
max_num = 1000
# Creamos la figura y el eje
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # espacio para el slider

# Dibujamos la primera secuencia
v = collatz_sequence(numero_inicial)
[line] = ax.plot(v, lw=2)
ax.set_title(f"Secuencia de Collatz para {numero_inicial}")

# Creamos el slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
slider = Slider(ax_slider, 'Número', valmin=min_num, valmax=max_num, valinit=numero_inicial, valstep=1)

# Función que se llama cuando se mueve el slider
def update(val):
    numero = int(slider.val)
    v = collatz_sequence(numero)
    line.set_ydata(v)
    line.set_xdata(range(len(v)))
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"Secuencia de Collatz para {numero}")
    fig.canvas.draw_idle()

# Conectamos el slider con la función
slider.on_changed(update)

# Función para controlar con teclas
def key_press(event):
    step = 1
    if event.key == "left":
        slider.set_val(max(slider.val - step, min_num))
    elif event.key == "right":
        slider.set_val(min(slider.val + step, max_num))

# Conectar eventos de teclado
fig.canvas.mpl_connect("key_press_event", key_press)

plt.show()
