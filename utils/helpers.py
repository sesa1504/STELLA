from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple
import numpy as np

class EventDataHelper:
    @staticmethod
    def extract_event_points(
        file_ext: str, data: Any, file_path: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        if data is None or not file_ext:
            return None
        file_ext = file_ext.lower()
        
        if file_ext in (".raw") and isinstance(data, dict):
            return EventDataHelper._events_from_mat(data)
        
        if file_ext == ".npy":
            return EventDataHelper._events_from_numpy_array(data)
        if file_ext == ".npz":
            return EventDataHelper._events_from_mat(data)
        if file_ext == ".mat":
            return EventDataHelper._events_from_mat(data)
        return None

    @staticmethod
    def _events_from_numpy_array(
        array: Any,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        if not isinstance(array, np.ndarray) or array.size == 0:
            return None

        arr = np.squeeze(array)
        if arr.size == 0:
            return None

        xs = ys = ts = None
        ps = None

        if arr.dtype.names:
            names = list(arr.dtype.names)
            x_field = EventDataHelper._match_field(names, ("x",))
            y_field = EventDataHelper._match_field(names, ("y",))
            t_field = EventDataHelper._match_field(names, ("t", "T", "ts", "time", "timestamp", "timestamps"))
            p_field = EventDataHelper._match_field(names, ("p", "pol", "polarity"))

            if x_field and y_field and t_field:
                xs = np.array(arr[x_field]).flatten()
                ys = np.array(arr[y_field]).flatten()
                ts = np.array(arr[t_field]).flatten()
                ps = np.array(arr[p_field]).flatten() if p_field else None

        elif arr.ndim >= 2 and arr.shape[-1] >= 3:
            flattened = arr.reshape(-1, arr.shape[-1])

            xs = flattened[:, 0]
            ys = flattened[:, 1]

            remaining_cols = list(range(2, flattened.shape[1]))

            pol_idx = None
            for j in reversed(remaining_cols):  
                candidate = flattened[:, j]
                uniq = np.unique(candidate[~np.isnan(candidate)] if np.issubdtype(candidate.dtype, np.floating) else candidate)
                if uniq.size > 0 and np.isin(uniq, [-1, 0, 1]).all():
                    pol_idx = j
                    break

            t_idx = None
            for j in remaining_cols:
                if j == pol_idx:
                    continue
                t_idx = j
                break

            if t_idx is not None:
                ts = flattened[:, t_idx]

            if pol_idx is not None:
                ps = flattened[:, pol_idx]

        if xs is None or ys is None or ts is None:
            return None

        xs = np.asarray(xs).flatten()
        ys = np.asarray(ys).flatten()
        ts = np.asarray(ts).flatten()
        size = min(xs.size, ys.size, ts.size)
        xs = xs[:size]
        ys = ys[:size]
        ts = ts[:size]

        if ps is not None:
            ps = np.asarray(ps).flatten()
            if ps.size != size:
                ps = ps[:size]

        return xs, ys, ts, ps
    
    @staticmethod
    def _events_from_mat(
        mat_data: Any,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        if not isinstance(mat_data, dict):
            return None

        cleaned = {k: v for k, v in mat_data.items() if not k.startswith("__")}
        if not cleaned:
            return None

        names = list(cleaned.keys())
        x_key = EventDataHelper._match_field(names, ("x", "X"))
        y_key = EventDataHelper._match_field(names, ("y", "Y"))
        t_key = EventDataHelper._match_field(names, ("t", "T", "ts", "time", "timestamp", "timestamps"))
        p_key = EventDataHelper._match_field(names, ("p", "P", "pol", "polarity"))

        if x_key and y_key and t_key:
            xs = EventDataHelper._flatten_numeric_array(cleaned[x_key])
            ys = EventDataHelper._flatten_numeric_array(cleaned[y_key])
            ts = EventDataHelper._flatten_numeric_array(cleaned[t_key])

            if xs is not None and ys is not None and ts is not None:
                size = min(xs.size, ys.size, ts.size)
                xs = xs[:size]
                ys = ys[:size]
                ts = ts[:size]

                ps = None
                if p_key:
                    ps = EventDataHelper._flatten_numeric_array(cleaned[p_key])
                    if ps is not None and ps.size != size:
                        ps = ps[:size]

                return xs, ys, ts, ps

        for value in cleaned.values():
            result = EventDataHelper._events_from_numpy_array(np.array(value))
            if result is not None:
                return result

        return None

    @staticmethod
    def _match_field(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
        lowercase_names = {name.lower(): name for name in names}
        for candidate in candidates:
            if candidate in lowercase_names:
                return lowercase_names[candidate]
        for name in names:
            for candidate in candidates:
                if name.lower().startswith(candidate):
                    return name
        return None

    @staticmethod
    def _flatten_numeric_array(value: Any) -> Optional[np.ndarray]:
        try:
            arr = np.array(value, dtype=np.float32).ravel()
        except Exception:
            try:
                arr = np.array(value).astype(np.float32, copy=False).ravel()
            except Exception:
                return None
        if arr.size == 0:
            return None
        return arr