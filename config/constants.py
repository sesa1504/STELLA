from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

APP_TITLE: str = "STELLA - v1.0.2"
VIEWPORT_SIZE: Tuple[int, int] = (1920, 1030)

DEFAULT_PARAMS: Dict[str, float | int | bool] = {
    "epsilon": 4.0,
    "range": 6.0,
    "minPts": 7,
    "Lmin": 7,
    "v_max": 3000,
    "dt": 1000,
    "height": 720,
    "width": 1280,
    "buffer": 5,
    "max_tracks": 100000,
    "max_length": 100000,
    "search_factor": 5,
    "steps_rev": 10000,
    "overlap": 0.4,
    "multitimefactor": 2,
    "use_only_pos": True,
    "live_view": False,
    "find_simple": False,
    "pixelwise_extension": True,
    "db_clustering": False,
    "correction": False,
    "correct_tracks": False,
    "lightweight_mode": False,
    "pseudo_images": False,
    "save_it": False,
    "spline_fitting": True,
    "Kalman_afterwards": True,
    "use_both": False,
    "chopperwheel": False,
    "multirun": False,
}

SUPPORTED_FORMATS: Sequence[str] = (
    ".npz",
    ".mat",
    ".raw",
)

PSEUDOFAME_DIMENSIONS: Tuple[int, int] = (1280, 720)

THEME_ROUNDING: Dict[int, float] = {
    0: 10.0,  # Frame rounding
    1: 10.0,  # Child rounding
    2: 10.0,  # Window rounding
    3: 10.0,  # Popup rounding
}

THEME_COLORS: Dict[str, Tuple[int, int, int]] = {
    "button": (70, 130, 180),
    "button_hovered": (90, 150, 200),
    "button_active": (50, 110, 160),
    "header": (60, 120, 170),
    "header_hovered": (80, 140, 190),
    "header_active": (40, 100, 150),
    "frame_bg": (40, 40, 40),
    "window_bg": (30, 30, 30),
    "child_bg": (25, 25, 25),
}


@dataclass(frozen=True)
class TabDefinition:
    key: str
    label: str


TAB_DEFINITIONS: Sequence[TabDefinition] = (
    TabDefinition("home", "Preprocessing"),
    TabDefinition("detection", "Detection"),
    TabDefinition("tracking", "Tracking"),
    TabDefinition("validation", "Validation"),
)
