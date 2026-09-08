import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm

def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    C = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    C = C[:, 0] + 1j * C[:, 1]

    fractal = np.zeros(len(C), dtype=int)

    for i, c in tqdm(enumerate(C), total=len(C), desc="Calculando Mandelbrot"):
        fractal[i] = mandelbrot(c, max_iter)

    return fractal.reshape((height, width))

class MandelbrotPlot:
    def __init__(self, fig, ax, width=250, height=250):
        self.fig = fig
        self.ax = ax
        self.width = width
        self.height = height
        self.updating = False  # Evitar recursión infinita
        self.max_iter = 500

        # Colorbar y referencia a la imagen
        self.img = None
        self.cbar = None

        # Conectar eventos de zoom/pan
        self._xlim_cid = self.ax.callbacks.connect('xlim_changed', self.on_zoom_pan)
        self._ylim_cid = self.ax.callbacks.connect('ylim_changed', self.on_zoom_pan)

        self.update_fractal(-2, 1, -1.5, 1.5)

    def update_fractal(self, xmin, xmax, ymin, ymax):
        self.updating = True

        fractal = compute_mandelbrot(xmin, xmax, ymin, ymax, self.width, self.height, self.max_iter)

        if self.img is None:
            # Crear la imagen y la barra de colores la primera vez
            self.img = self.ax.imshow(fractal.T, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='viridis')

            self.cbar = self.fig.colorbar(self.img, ax=self.ax, orientation='vertical')
            self.cbar.set_label('Iteraciones')
        else:
            # Solo actualizar datos, extensión y colorbar en sucesivos zooms
            self.img.set_data(fractal.T)
            self.img.set_extent([xmin, xmax, ymin, ymax])
            self.img.set_clim(vmin=fractal.min(), vmax=fractal.max())

            self.cbar.update_normal(self.img)  # <- Actualiza la barra de colores

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.figure.canvas.draw()

        self.updating = False

    def on_zoom_pan(self, event_ax):
        if self.updating or event_ax != self.ax:
            return

        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        self.update_fractal(xmin, xmax, ymin, ymax)

    def set_max_iter(self, val):
        self.max_iter = int(val)
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        self.update_fractal(xmin, xmax, ymin, ymax)

def main():
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)

    mandelbrot_plot = MandelbrotPlot(fig, ax)

    # Slider para el número de iteraciones
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgray')
    slider = Slider(ax_slider, 'Iteraciones', 100, 1000, valinit=500, valstep=50)

    slider.on_changed(mandelbrot_plot.set_max_iter)

    plt.show()

if __name__ == '__main__':
    main()
