from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union
import sys

class TrackingCancelled(Exception):
    """Raised when a tracking run is cancelled (optional, for future UI wiring)."""
    pass

@dataclass
class TrackingOutputs:
    x_plot: Any
    y_plot: Any
    x_plotv: Any
    y_plotv: Any
    t_plot: Any
    track_to_cluster: Any

class TrackingEngine:
    def __init__(self, script_path: Union[str, Path]):
        self.script_path = Path(script_path)
        if not self.script_path.exists():
            raise FileNotFoundError(f"Tracking script not found: {self.script_path}")

        self._script_text = self.script_path.read_text(encoding="utf-8")
        self._compiled = compile(self._script_text, str(self.script_path), "exec")

    def run(self, context: Dict[str, Any]) -> TrackingOutputs:
        ctx: Dict[str, Any] = dict(context)

        ctx.setdefault("__file__", str(self.script_path))
        ctx.setdefault("__name__", "__tracking_exec__")

        script_dir = str(self.script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        exec(self._compiled, ctx, ctx)

        return TrackingOutputs(
            x_plot=ctx["x_plot"],
            y_plot=ctx["y_plot"],
            x_plotv=ctx["x_plotv"],
            y_plotv=ctx["y_plotv"],
            t_plot=ctx["t_plot"],
            track_to_cluster=ctx.get("track_to_cluster", None),
        )