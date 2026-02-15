from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from metavision_core.event_io import EventsIterator
    METAVISION_AVAILABLE = True
except Exception:
    EventsIterator = None
    METAVISION_AVAILABLE = False

@dataclass
class LoadedData:
    file_path: str
    file_ext: str
    data: Any
    status_message: str

    width: Optional[int] = None
    height: Optional[int] = None
    openeb_info: Optional[dict[str, Any]] = None

def _load_events_with_metavision(
    file_path: str, delta_t: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[int], Optional[int]]:

    if not METAVISION_AVAILABLE or EventsIterator is None:
        raise ImportError("OpenEB not available (cannot import EventsIterator).")

    mv_iterator = EventsIterator(input_path=file_path, delta_t=delta_t)

    width: Optional[int] = None
    height: Optional[int] = None
    try:
        h, w = mv_iterator.get_size()
        height, width = int(h), int(w)
    except Exception:
        pass

    xs_chunks: list[np.ndarray] = []
    ys_chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []
    ps_chunks: list[np.ndarray] = []

    for evs in mv_iterator:
        xs_chunks.append(np.asarray(evs["x"]))
        ys_chunks.append(np.asarray(evs["y"]))
        ts_chunks.append(np.asarray(evs["t"]))
        ps_chunks.append(np.asarray(evs["p"]))

    if not xs_chunks:
        x = np.array([], dtype=np.int16)
        y = np.array([], dtype=np.int16)
        t = np.array([], dtype=np.int64)
        p = np.array([], dtype=np.int8)
    else:
        x = np.concatenate(xs_chunks)
        y = np.concatenate(ys_chunks)
        t = np.concatenate(ts_chunks)
        p = np.concatenate(ps_chunks)

    return x, y, t, p, width, height


def load_data_file(file_path: str) -> LoadedData:
    path = Path(file_path)
    file_ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(file_path)

    data: Any = None
    status = ""
    width: Optional[int] = None
    height: Optional[int] = None
    openeb_info: Optional[dict[str, Any]] = None

    if file_ext == ".mat":
        from scipy.io import loadmat

        data = loadmat(file_path)
        status = "Valid .mat file."

    elif file_ext == ".npy":
        data = np.load(file_path, allow_pickle=True)
        status = f"Valid .npy file. Shape: {getattr(data, 'shape', None)}"
        
    elif file_ext == ".npz":
        with np.load(file_path, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        status = f"Valid .npz file. Keys: {list(data.keys())[:8]}"

    elif file_ext in (".raw", ".dat"):
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        if not METAVISION_AVAILABLE:
            status = f"Valid {file_ext} file. Size: {size_mb:.2f} MB (Metavision SDK NOT available)"
        else:
            try:
                x, y, t, p, w, h = _load_events_with_metavision(file_path, delta_t=1000)
                data = {"x": x, "y": y, "ts": t, "p": p}
                width, height = w, h
                status = f"Loaded {file_ext} via OpenEB. Size: {size_mb:.2f} MB"
                if width is not None and height is not None:
                    status += f", Resolution: {width}x{height}"
                status += f", Events: {int(x.size):,}"
            except Exception as exc:
                status = f"Error loading {file_ext} via OpenEB: {exc}"
    elif file_ext in (".bin", ".txt"):
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        status = f"Valid {file_ext} file. Size: {size_mb:.2f} MB"
    else:
        status = f"Unknown format: {file_ext}"

    return LoadedData(
        file_path=file_path,
        file_ext=file_ext,
        data=data,
        status_message=status,
        width=width,
        height=height,
        openeb_info=openeb_info,
    )


def build_preview_and_stats(loaded: LoadedData) -> tuple[str, str]:
    file_path = loaded.file_path
    file_ext = loaded.file_ext
    data = loaded.data

    preview_text = f"File: {Path(file_path).name}\n"
    preview_text += f"Format: {file_ext}\n"
    preview_text += f"Size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB\n\n"

    if file_ext == ".mat" and isinstance(data, dict):
        preview_text += "Contents:\n"
        for key in list(data.keys())[:10]:
            if key.startswith("__"):
                continue
            value = data[key]
            if hasattr(value, "shape"):
                preview_text += f"  {key}: {value.shape} {getattr(value, 'dtype', '')}\n"
            else:
                preview_text += f"  {key}: {type(value).__name__}\n"

    elif file_ext == ".npy" and isinstance(data, np.ndarray):
        preview_text += f"Shape: {data.shape}\n"
        preview_text += f"Type: {data.dtype}\n"
        preview_text += f"Sample:\n{str(data.flat[:5])}\n"
        
    elif file_ext == ".npz" and isinstance(data, dict):
        preview_text += "Contents:\n"
        for key in list(data.keys())[:10]:
            value = data[key]
            if hasattr(value, "shape"):
                preview_text += f"  {key}: {value.shape} {getattr(value, 'dtype', '')}\n"
            else:
                preview_text += f"  {key}: {type(value).__name__}\n"
    return preview_text
