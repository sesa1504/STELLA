from __future__ import annotations
import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use("Agg")

def figure_to_rgba_flat(fig) -> np.ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    rgba = (buf.astype(np.float32) / 255.0)
    return rgba.flatten()