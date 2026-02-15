from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import sys

class DetectionCancelled(Exception):
    pass

@dataclass
class DetectionOutputs:
    x_clust: Any
    y_clust: Any
    t_clust: Any
    steps: Any
    x_m: Any
    y_m: Any
    t_m: Any
    ind_clust: Any


class DetectionEngine:
    def __init__(self, script_path: Union[str, Path]):
        self.script_path = Path(script_path)
        if not self.script_path.exists():
            raise FileNotFoundError(f"Detection script not found: {self.script_path}")

        self._script_text = self.script_path.read_text(encoding="utf-8")
        self._compiled = compile(self._script_text, str(self.script_path), "exec")
        

    def run(self, context: Dict[str, Any]) -> DetectionOutputs:
        ctx: Dict[str, Any] = dict(context)

        ctx.setdefault("__file__", str(self.script_path))
        ctx.setdefault("__name__", "__detection_exec__")

        script_dir = str(self.script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        exec(self._compiled, ctx, ctx)

        return DetectionOutputs(
            x_clust=ctx["x_clust"],
            y_clust=ctx["y_clust"],
            t_clust=ctx["t_clust"],
            steps=ctx["steps"],
            x_m=ctx["x_m"],
            y_m=ctx["y_m"],
            t_m=ctx["t_m"],
            ind_clust=ctx["ind_clust"],
        )
    
