import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.io as sio
except ImportError:
    sio = None


def load_saved_result(path: str):
    """Load saved .npz or .mat that contains result_pickle/ctx_pickle."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        result_bytes = data["result_pickle"].tobytes()
        ctx_bytes = data["ctx_pickle"].tobytes()
        meta = data["meta"][0] if "meta" in data else None

    elif suffix == ".mat":
        if sio is None:
            raise RuntimeError("scipy is required to load .mat files (pip install scipy).")
        data = sio.loadmat(p)
        result_bytes = data["result_pickle"].tobytes()
        ctx_bytes = data["ctx_pickle"].tobytes()
        meta = data.get("meta", None)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    result = pickle.loads(result_bytes)
    ctx = pickle.loads(ctx_bytes)
    return result, ctx, meta


def get_field(obj, name):
    """Works for dict or attribute objects."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def plot_tracks_in_time_window(result, t_start, t_end, title="Tracks in time window",
                               max_tracks=None, min_points=2):
    """
    Plot all track parts whose timestamps t are within [t_start, t_end].

    - t_start/t_end: start/end time in us
    - max_tracks: optional cap (None = no cap)
    - min_points: ignore very short snippets
    """
    x_plot = get_field(result, "x_plot")
    y_plot = get_field(result, "y_plot")
    t_plot = get_field(result, "t_plot")

    if x_plot is None or y_plot is None or t_plot is None:
        raise ValueError("result is missing x_plot / y_plot / t_plot")

    # ensure list-of-tracks
    if not isinstance(x_plot, (list, tuple)): x_plot = [x_plot]
    if not isinstance(y_plot, (list, tuple)): y_plot = [y_plot]
    if not isinstance(t_plot, (list, tuple)): t_plot = [t_plot]

    n_tracks = min(len(x_plot), len(y_plot), len(t_plot))

    plt.figure()

    plotted = 0
    for i in range(n_tracks):
        if max_tracks is not None and plotted >= max_tracks:
            break

        x = np.asarray(x_plot[i], dtype=float)
        y = np.asarray(y_plot[i], dtype=float)
        t = np.asarray(t_plot[i], dtype=float)

        # basic sanity
        m = min(len(x), len(y), len(t))
        x, y, t = x[:m], y[:m], t[:m]

        # keep only points inside time window
        mask = (t >= t_start) & (t <= t_end) & np.isfinite(t)
        if mask.sum() < min_points:
            continue

        xw = x.copy()
        yw = y.copy()
        xw[~mask] = np.nan
        yw[~mask] = np.nan

        plt.plot(xw, yw, linewidth=1, c="g")
        plotted += 1

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"{title} (t∈[{t_start}, {t_end}])  showing {plotted}/{n_tracks}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    # change this:
    path = "[path to validated tracks]" # .npz or .mat
    result, ctx, meta = load_saved_result(path)
    
    t_plot = get_field(result, "t_plot")
    t_min = min(
            np.nanmin(t)
            for t in t_plot
            if len(t) > 0 and np.isfinite(t).any()
        )
    
    acc_time = 10000

    plot_tracks_in_time_window(result, t_start=t_min, t_end=t_min+acc_time,
                               title="Validated tracks", max_tracks=None)
