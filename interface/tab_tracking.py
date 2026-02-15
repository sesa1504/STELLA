from __future__ import annotations
from copy import deepcopy
from typing import Any, Optional, TYPE_CHECKING
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from utils.plots import figure_to_rgba_flat
import dearpygui.dearpygui as dpg
import numpy as np
import pickle
import scipy.io as sio
from pathlib import Path
from datetime import datetime
import sys, io, contextlib, threading, queue

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tracking_engine import TrackingEngine, TrackingCancelled
from config.constants import DEFAULT_PARAMS, PSEUDOFAME_DIMENSIONS

if TYPE_CHECKING:
    from interface.tab_preprocessing import PreprocessingTab
    from interface.tab_detection import DetectionTab


class QueueWriter(io.TextIOBase):
    def __init__(self, q: queue.Queue[str]):
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self.q.put(self._buf)
        self._buf = ""


class TrackingTab:
    label: str = "Tracking"

    def __init__(
        self,
        preprocessing_tab: Optional[PreprocessingTab] = None,
        detection_tab: Optional[DetectionTab] = None,
    ) -> None:
        self.preprocessing_tab = preprocessing_tab
        self.detection_tab = detection_tab

        self.params: dict[str, Any] = deepcopy(DEFAULT_PARAMS)
        self.params.setdefault("num_worker", 8)
        self.params.setdefault("resol", 1.0)
        self.params.setdefault("target_RMS", 3.0)
        self.params.setdefault("beta", 0.1)
        
        self.params.setdefault("Q_scale", 1e-3)
        self.params.setdefault("R", 1.0)
        self.params.setdefault("use_hybrid", False)
        self.params.setdefault("both2", False)

        script_path = Path(__file__).resolve().parents[1] / "utils" / "tracking.py"
        self.engine = TrackingEngine(script_path)
        self._cluster_sorted_idx: list[int] = []
        self._cluster_pos: int = 0
        self._cluster_signature: object = None
        
        self._cluster_series_xy_tag = "track_cluster_scatter_xy"
        self._cluster_series_xt_tag = "track_cluster_scatter_xt"
        self._cluster_series_yt_tag = "track_cluster_scatter_yt"
        
        self._cluster_xy_x_axis_tag = "track_cluster_xy_x_axis"
        self._cluster_xy_y_axis_tag = "track_cluster_xy_y_axis"
        self._cluster_xt_x_axis_tag = "track_cluster_xt_x_axis"
        self._cluster_xt_y_axis_tag = "track_cluster_xt_y_axis"
        self._cluster_yt_x_axis_tag = "track_cluster_yt_x_axis"
        self._cluster_yt_y_axis_tag = "track_cluster_yt_y_axis"
        
        self._track_line_xy_tag = "track_line_xy"
        self._track_line_xt_tag = "track_line_xt"
        self._track_line_yt_tag = "track_line_yt" 

        self._last_test_cluster_idx: Optional[int] = None
        
        self._cluster_info_tag = "track_cluster_info"

        self._track_method_combo_tag = "track_method_combo"
        self._track_method_options = [
            "Kalman filtering",
            "B-spline",
            "hybrid(xk,us)",
            "hybrid(xk,uks)",
            "hybrid(xks,uks)",
        ]

        self._run_status_tag = "track_run_status"
        self._run_button_tag = "track_run_button"
        
        self._group_kalman_params = "track_group_kalman_params"

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._run_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_run_ctx: Optional[dict[str, Any]] = None
        self._track_run_stats_text_tag = "track_run_stats"
        
        self.pseudoframe_dimensions = PSEUDOFAME_DIMENSIONS
        self._pf_uv_min = [0.0, 0.0]
        self._pf_uv_max = [1.0, 1.0]
        
        self._pf_status_tag = "track_pseudoframe_status"
        self._pf_time_info_tag = "track_pseudoframe_time_info"
        self._pf_drawlist_tag = "track_pf_drawlist"
        self._pf_drawimage_tag = "track_pf_drawimage"
        self._pf_tracks_node_tag = "track_pf_tracks_node"
        
        self.params.setdefault("track_vmax", 0.0)  
        self._pf_vmax_input_tag = "track_pf_vmax"
        
        self.duration_hist_texture_tag = "track_duration_hist_texture"
        self._texture_registry_tag = "track_texture_registry"

        self.params.setdefault("track_timefactor", 2.0)
        self._pf_timefactor_input_tag = "track_pf_timefactor"
        self.last_result = None  

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def build(self, parent: int) -> None:
        self._ensure_textures()
        with dpg.group(parent=parent):
            with dpg.group(horizontal=True):
                with dpg.child_window(label="Tracking", width=500, height=950):
                    self._build_data_panel()

                with dpg.child_window(label="Visualization", width=1920 - 550, height=950):
                    dpg.add_text("VISUALIZATION", color=(100, 200, 255))
                    dpg.add_separator()

                    with dpg.tab_bar(tag="track_vis_tabbar"):
                        with dpg.tab(label="Clusters"):
                            self._build_cluster_plot_panel()
                        with dpg.tab(label="Pseudo-frame"):
                            self._build_tracking_pseudoframe_panel()
                        with dpg.tab(label="Track duration", tag="track_vis_tab_duration_hist"):
                            self._build_track_duration_hist_panel()
        self._status_log_append("Ready.")
        
    def _ensure_textures(self) -> None:
        registry_tag = "texture_registry_main"
    
        if not dpg.does_item_exist(self._texture_registry_tag):
            dpg.add_texture_registry(tag=self._texture_registry_tag)
    
        if not dpg.does_item_exist(registry_tag):
            dpg.add_texture_registry(tag=registry_tag)
    
        if not dpg.does_item_exist(self.duration_hist_texture_tag):
            dummy = np.zeros((580, 700, 4), dtype=np.float32)
            dpg.add_raw_texture(
                width=700,
                height=580,
                default_value=dummy.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag=self.duration_hist_texture_tag,
                parent=registry_tag,
            )

    def _build_data_panel(self) -> None:
        with dpg.theme() as info_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 150, 200))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 110, 160))
        
                dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 130, 180, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
        
        dpg.add_text("TRACKING", color=(100, 200, 255))
        dpg.add_separator()

        dpg.add_text("Tracking method:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_combo(
                items=self._track_method_options,
                default_value=self.params.get("tracking_method", self._track_method_options[0]),
                tag=self._track_method_combo_tag,
                width=400,
                callback=self._on_track_method_change,
            )
            dpg.add_button(label="?", width=22, height=22, tag="track_info_btn")
            dpg.bind_item_theme("track_info_btn", info_button_theme)
        
            with dpg.popup("track_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "Kalman filtering:\n"
                    "    Position: Kalman, Velocity: Kalman\n"
                    "B-spline:\n"
                    "    Position: spline, Velocity: spline\n"
                    "hybrid(xk,us):\n"
                    "    Position: Kalman, Velocity: spline\n"
                    "hybrid(xk,uks):\n"
                    "    Position: Kalman, Velocity: Kalman+spline\n"
                    "hybrid(xks,uks):\n"
                    "    Position: Kalman+spline, Velocity: Kalman+spline"
                )
        dpg.add_separator()

        dpg.add_text("Parameter", color=(200, 200, 100))
        
        
        current_method = self.params.get("tracking_method", self._track_method_options[0])
        with dpg.group(tag=self._group_kalman_params, show=(current_method == "Kalman filtering")):
            dpg.add_input_int(
                label="number of workers",
                default_value=int(self.params.get("num_worker", 8)),
                width=200,
                tag="track_input_num_worker",
                callback=lambda s, a, u: self._set_param("num_worker", int(a)),
            )
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="P",
                    default_value=float(self.params.get("P", 1e8)),
                    width=200,
                    tag="track_input_P",
                    callback=lambda s, a, u: self._set_param("P", float(a)),
                )
                dpg.add_button(label="?", width=22, height=22, tag="P_info_btn")
                dpg.bind_item_theme("P_info_btn", info_button_theme)
            
                with dpg.popup("P_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "Initial model uncertainty."
                    )
            with dpg.group(horizontal=True):
                dpg.add_text("Q matrix")
                dpg.add_button(label="?", width=22, height=22, tag="Q_info_btn")
                dpg.bind_item_theme("Q_info_btn", info_button_theme)
            
                with dpg.popup("Q_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "Process noise covariance matrix Q,\n"
                        "which quantifies the uncertainty\n"
                        "in the underlying motion model.\n"
                        "As Q is increased, the Kalman filter\n"
                        "will respont more aggressively to\n"
                        "changes in position."
                    )
            
            self.q_inputs = []
            
            default = np.diag([2e-6, 1e0, 4e5]).astype(np.float64)
            for i in range(3):
                with dpg.group(horizontal=True):
                    row = []
                    for j in range(3):
                        row.append(
                            dpg.add_input_float(
                                label=f"##Q_{i}_{j}",
                                default_value=float(default[i, j]),
                                width=150,
                                format="%.1e"
                            )
                        )
                    self.q_inputs.append(row)
            
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="R",
                    default_value=float(self.params.get("R", 5.0)),
                    width=200,
                    tag="track_input_R",
                    callback=lambda s, a, u: self._set_param("R", float(a)),
                )
                dpg.add_button(label="?", width=22, height=22, tag="R_info_btn")
                dpg.bind_item_theme("R_info_btn", info_button_theme)
            
                with dpg.popup("R_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "Measurement noise covariance R,\n"
                        "which quantifies the uncertainty\n"
                        "in the measured values."
                    )
            
        with dpg.group(show=False, tag="track_group_bspline_params"):
            with dpg.group(horizontal=True):
                self.target_rms = dpg.add_input_float(
                    label="target RMS",
                    default_value=float(self.params.get("target_RMS", 3.0)),
                    width=200,
                    format="%.3f"
                )
                dpg.add_button(label="?", width=22, height=22, tag="RMS_info_btn")
                dpg.bind_item_theme("RMS_info_btn", info_button_theme)
            
                with dpg.popup("RMS_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "In an iterative scheme, the\n"
                        "smoothing of calculated splines\n"
                        "is reduced, until the target RMS\n"
                        "to the data points is reached."
                    )
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="resolution factor",
                default_value=float(self.params.get("resol", 1.0)),
                width=200,
                tag="track_input_resol",
                callback=lambda s, a, u: self._set_param("resol", float(a)),
            )
            dpg.add_button(label="?", width=22, height=22, tag="res_info_btn")
            dpg.bind_item_theme("res_info_btn", info_button_theme)
        
            with dpg.popup("res_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "Quantifies the number of track points\n"
                    "calculated within one time step,\n"
                    "i.e. the track resolution."
                )

        self.apply_correction = dpg.add_checkbox(
            label="apply offset correction",
            default_value=False,
            callback=self._on_apply_correction_change
        )
        
        with dpg.group(show=False, tag="track_group_beta"):
            with dpg.group(horizontal=True):
                self.beta = dpg.add_input_float(
                    label="beta",
                    default_value=0.8,
                    width=200,
                    format="%.3f"
                )
                dpg.add_button(label="?", width=22, height=22, tag="beta_info_btn")
                dpg.bind_item_theme("beta_info_btn", info_button_theme)
            
                with dpg.popup("beta_info_btn", mousebutton=dpg.mvMouseButton_Left):
                    dpg.add_text(
                        "Scaling factor for the offset\n"
                        "correction."
                    )
        dpg.add_separator()
        
        dpg.add_button(
            label="Test tracking",
            tag="track_test_button",
            width=400,
            callback=self._on_test_tracking_clicked,
        )
        
        dpg.add_separator()

        dpg.add_button(
            label="Run tracking",
            tag=self._run_button_tag,
            width=400,
            callback=self._on_run_tracking_clicked,
        )
        
        dpg.add_button(
            label="Clear results",
            tag="track_clear_results_button",
            width=200,
            callback=self._on_clear_tracking_results_clicked,
        )
        
        dpg.add_text("", tag=self._run_status_tag, wrap=400, color=(100, 255, 100))
        
        dpg.add_text("Tracking stats:", color=(200, 200, 100))
        with dpg.child_window(
            tag="track_run_stats_box",
            height=110,
            autosize_x=True,
            border=True,
        ):
            dpg.add_text("No results yet.", tag=self._track_run_stats_text_tag, wrap=430)
            
        dpg.add_separator()
        dpg.add_text("EXPORT / IMPORT RESULTS", color=(100, 200, 255))
        dpg.add_text("Save tracking results:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="track_save_results_file_input", width=300, hint="Save as .npz or .mat")
            dpg.add_button(label="Browse", callback=self.browse_track_save_results_file, width=80)
            dpg.add_combo(items=[".npz", ".mat"], default_value=".npz", tag="track_save_results_format", width=70)
        dpg.add_button(label="Save results", callback=self.save_tracking_results, width=120)
        dpg.add_text("", tag="track_save_results_status", wrap=430, color=(100, 255, 100))
        
        dpg.add_text("Load tracking results:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="track_load_results_file_input", width=300, hint="Load .npz or .mat")
            dpg.add_button(label="Browse", callback=self.browse_track_load_results_file, width=80)
        dpg.add_button(label="Load results", callback=self.load_tracking_results, width=120)
        dpg.add_text("", tag="track_load_results_status", wrap=430, color=(100, 255, 100))
        
        if not dpg.does_item_exist("track_save_results_file_dialog"):
            with dpg.file_dialog(
                tag="track_save_results_file_dialog",
                label="Save tracking results",
                directory_selector=False,
                show=False,
                callback=self.on_track_save_results_file_selected,
                cancel_callback=lambda s, a: dpg.set_value("track_save_results_status", "Save cancelled"),
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))
        
        if not dpg.does_item_exist("track_load_results_file_dialog"):
            with dpg.file_dialog(
                tag="track_load_results_file_dialog",
                label="Load tracking results",
                directory_selector=False,
                show=False,
                callback=self.on_track_load_results_file_selected,
                cancel_callback=lambda s, a: dpg.set_value("track_load_results_status", "Load cancelled"),
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))
        
    def _build_cluster_plot_panel(self) -> None:
        with dpg.group(horizontal=True):
            with dpg.plot(height=400, width=400):
                dpg.add_plot_axis(dpg.mvXAxis, label="x in pixel", tag=self._cluster_xy_x_axis_tag)
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="y in pixel", tag=self._cluster_xy_y_axis_tag)
                dpg.add_scatter_series([], [], label="cluster", parent=y_axis, tag=self._cluster_series_xy_tag)
                dpg.add_line_series([], [], label="tracking", parent=y_axis, tag=self._track_line_xy_tag)

            with dpg.plot(height=400, width=400):
                dpg.add_plot_axis(dpg.mvXAxis, label="t in s", tag=self._cluster_xt_x_axis_tag)
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="x in pixel", tag=self._cluster_xt_y_axis_tag)
                dpg.add_scatter_series([], [], label="cluster", parent=y_axis, tag=self._cluster_series_xt_tag)
                dpg.add_line_series([], [], label="tracking", parent=y_axis, tag=self._track_line_xt_tag)

            with dpg.plot(height=400, width=400):
                dpg.add_plot_axis(dpg.mvXAxis, label="t in s", tag=self._cluster_yt_x_axis_tag)
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="y in pixel", tag=self._cluster_yt_y_axis_tag)
                dpg.add_scatter_series([], [], label="cluster", parent=y_axis, tag=self._cluster_series_yt_tag)
                dpg.add_line_series([], [], label="tracking", parent=y_axis, tag=self._track_line_yt_tag)
    
        dpg.add_spacer(height=5)
    
        with dpg.group(horizontal=True):
            dpg.add_button(label="Prev", width=80, callback=self._on_prev_cluster)
            dpg.add_button(label="Next", width=80, callback=self._on_next_cluster)
            dpg.add_spacer(width=10)
            dpg.add_text("", tag=self._cluster_info_tag, wrap=800, color=(100, 255, 100))
    
        self._refresh_cluster_order_if_needed(force=True)
        self._update_cluster_plot()
    
        dpg.add_separator()
    
        dpg.add_text("Status Log", color=(200, 200, 100))
        with dpg.child_window(label="Status Log", height=90, width=720):
            dpg.add_input_text(
                tag="track_status_log_text",
                multiline=True,
                readonly=True,
                width=695,
                height=75,
            )

    def _build_tracking_pseudoframe_panel(self) -> None:
        pseudo_texture = (
            self.preprocessing_tab.pseudoframe_texture_tag
            if self.preprocessing_tab is not None
            else "detection_pseudoframe_texture"  
        )
    
        w, h = self.pseudoframe_dimensions
    
        with dpg.drawlist(width=w, height=h, tag=self._pf_drawlist_tag):
            dpg.draw_image(
                pseudo_texture,
                pmin=[0, 0],
                pmax=[w, h],
                uv_min=self._pf_uv_min,
                uv_max=self._pf_uv_max,
                tag=self._pf_drawimage_tag,
            )
            with dpg.draw_node(tag=self._pf_tracks_node_tag):
                pass
    
        dpg.add_text("Load data in Preprocessing to generate pseudo-frame.", tag=self._pf_status_tag, wrap=730)
    
        with dpg.group(horizontal=True):
            dpg.add_button(label="<< Prev", width=120, callback=self._on_pf_prev)
            dpg.add_button(label="Next >>", width=120, callback=self._on_pf_next)
            dpg.add_text("", tag=self._pf_time_info_tag, wrap=600)
    
        if self.preprocessing_tab is not None:
            self.preprocessing_tab.register_pseudoframe_mirror(
                status_tag=self._pf_status_tag,
                time_info_tag=self._pf_time_info_tag,
            )
            
        dpg.add_input_float(
            label="Track window factor (× accumulation time)",
            default_value=float(self.params.get("track_timefactor", 2.0)),
            width=200,
            format="%.2f",
            on_enter=True,
            tag=self._pf_timefactor_input_tag,
            callback=lambda s, a, u: self._on_track_timefactor_change(float(a)),
        )
        
        dpg.add_input_float(
            label="Max velocity (plotting)",
            default_value=float(self.params.get("track_vmax", 0.0)),
            width=200,
            format="%.6f",
            on_enter=True,
            tag=self._pf_vmax_input_tag,
            callback=lambda s, a, u: self._on_track_vmax_change(float(a)),
        )
    
    def _build_track_duration_hist_panel(self):
        with dpg.child_window(
            tag="track_duration_hist_container",
            autosize_x=True,
            autosize_y=True,
            border=False,
        ):
            dpg.add_image(
                texture_tag=self.duration_hist_texture_tag,
                tag="track_duration_hist_image",
            )

    # ------------------------------------------------------------------
    # Cluster preview plotting (from detection output)
    # ------------------------------------------------------------------
    @staticmethod
    def _count_non_nan(arr) -> int:
        try:
            a = np.asarray(arr)
            if a.size == 0:
                return 0
            if np.issubdtype(a.dtype, np.number):
                return int(np.sum(~np.isnan(a)))
            return int(a.size)
        except Exception:
            return 0

    def _get_detection_clusters(self):
        if self.detection_tab is None:
            return None, None, None
        det_out = getattr(self.detection_tab, "last_result", None)
        if det_out is None:
            return None, None, None

        t_clust = getattr(det_out, "t_clust", None)
        x_clust = getattr(det_out, "x_clust", None)
        y_clust = getattr(det_out, "y_clust", None)
    
        if t_clust is None and isinstance(det_out, dict):
            t_clust = det_out.get("t_clust")
            x_clust = det_out.get("x_clust")
            y_clust = det_out.get("y_clust")
    
        if t_clust is None or x_clust is None or y_clust is None:
            return None, None, None
    
        return t_clust, x_clust, y_clust
    
    def _use_pseudo_frames_for_plot(self) -> bool:
        if self.detection_tab is None:
            return False

        det_method = getattr(self.detection_tab, "params", {}).get("detection_method", "pixelwise extension")
        return det_method == "pseudo-frame"
    
    def _get_detection_plot_series(self):
        if self.detection_tab is None:
            return None, None, None
        det_out = getattr(self.detection_tab, "last_result", None)
        if det_out is None:
            return None, None, None
    
        use_pseudo = self._use_pseudo_frames_for_plot()
    
        def _get(name: str):
            v = getattr(det_out, name, None)
            if v is None and isinstance(det_out, dict):
                v = det_out.get(name)
            return v
    
        if use_pseudo:
            t = _get("t_m")
            x = _get("x_m")
            y = _get("y_m")
        else:
            t = _get("t_clust")
            x = _get("x_clust")
            y = _get("y_clust")
    
        if t is None or x is None or y is None:
            return None, None, None
    
        return t, x, y

    def _refresh_cluster_order_if_needed(self, force: bool = False) -> None:
        det_out = getattr(self.detection_tab, "last_result", None) if self.detection_tab is not None else None
        signature = det_out
        if (not force) and (signature is self._cluster_signature):
            return

        self._cluster_signature = signature
        t_src, x_src, y_src = self._get_detection_plot_series()
        if t_src is None or x_src is None or y_src is None:
            self._cluster_sorted_idx = []
            self._cluster_pos = 0
            self._cluster_signature = None
            return

        sizes = []
        for i, xc in enumerate(x_src):
            sizes.append((self._count_non_nan(xc), i))
        sizes.sort(reverse=True, key=lambda p: p[0])
        self._cluster_sorted_idx = [i for _, i in sizes if _ > 0]
        if not self._cluster_sorted_idx:
            self._cluster_sorted_idx = [i for _, i in sizes]

        self._cluster_pos = 0
        self._update_cluster_plot()

    def _update_cluster_plot(self) -> None:
        needed = [self._cluster_series_xy_tag, self._cluster_series_xt_tag, self._cluster_series_yt_tag]
        if not all(dpg.does_item_exist(tag) for tag in needed):
            return
    
        t_src, x_src, y_src = self._get_detection_plot_series()
        if t_src is None or x_src is None or y_src is None or len(x_src) == 0:
            dpg.set_value(self._cluster_series_xy_tag, [[], []])
            dpg.set_value(self._cluster_series_xt_tag, [[], []])
            dpg.set_value(self._cluster_series_yt_tag, [[], []])
            if dpg.does_item_exist(self._cluster_info_tag):
                dpg.set_value(self._cluster_info_tag, "No detected series available.")
            return
        if not self._cluster_sorted_idx:
            self._cluster_sorted_idx = list(range(len(x_src)))
            self._cluster_pos = 0
    
        self._cluster_pos = max(0, min(self._cluster_pos, len(self._cluster_sorted_idx) - 1))
        i = self._cluster_sorted_idx[self._cluster_pos]
    
        try:
            t_all = np.asarray(t_src[i], dtype=float)
            x_all = np.asarray(x_src[i], dtype=float)
            y_all = np.asarray(y_src[i], dtype=float)
        except Exception:
            t_all, x_all, y_all = np.array([]), np.array([]), np.array([])
    
        def _set_limits(x_axis_tag: str, y_axis_tag: str, xs: np.ndarray, ys: np.ndarray) -> None:
            if len(xs) == 0 or len(ys) == 0:
                return
            if not (dpg.does_item_exist(x_axis_tag) and dpg.does_item_exist(y_axis_tag)):
                return
    
            x_min = float(np.min(xs)); x_max = float(np.max(xs))
            y_min = float(np.min(ys)); y_max = float(np.max(ys))
    
            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05
            if x_pad == 0.0: x_pad = 1.0
            if y_pad == 0.0: y_pad = 1.0
    
            dpg.set_axis_limits(x_axis_tag, x_min - x_pad, x_max + x_pad)
            dpg.set_axis_limits(y_axis_tag, y_min - y_pad, y_max + y_pad)
    
        mask_xy = (~np.isnan(x_all)) & (~np.isnan(y_all))
        x_xy = x_all[mask_xy]
        y_xy = y_all[mask_xy]
        dpg.set_value(self._cluster_series_xy_tag, [x_xy.tolist(), y_xy.tolist()])
        _set_limits(self._cluster_xy_x_axis_tag, self._cluster_xy_y_axis_tag, x_xy, y_xy)

        mask_xt = (~np.isnan(t_all)) & (~np.isnan(x_all))
        t_xt = t_all[mask_xt]*1e-6
        x_xt = x_all[mask_xt]
        dpg.set_value(self._cluster_series_xt_tag, [t_xt.tolist(), x_xt.tolist()])
        _set_limits(self._cluster_xt_x_axis_tag, self._cluster_xt_y_axis_tag, t_xt, x_xt)

        mask_yt = (~np.isnan(t_all)) & (~np.isnan(y_all))
        t_yt = t_all[mask_yt]*1e-6
        y_yt = y_all[mask_yt]
        dpg.set_value(self._cluster_series_yt_tag, [t_yt.tolist(), y_yt.tolist()])
        _set_limits(self._cluster_yt_x_axis_tag, self._cluster_yt_y_axis_tag, t_yt, y_yt)
    
        if dpg.does_item_exist(self._cluster_info_tag):
            n_show = max(len(x_xy), len(t_xt), len(t_yt))
            dpg.set_value(
                self._cluster_info_tag,
                f"Showing cluster index {i} (rank {self._cluster_pos+1}/{len(self._cluster_sorted_idx)}), n~{n_show}"
            )

    def _update_tracking_overlay(self) -> None:
        needed = [self._track_line_xy_tag, self._track_line_xt_tag, self._track_line_yt_tag]
        if not all(dpg.does_item_exist(tag) for tag in needed):
            return
        
        def _to_track_list(v):
            if v is None:
                return []
            if isinstance(v, np.ndarray):
                if v.ndim <= 1:
                    return [v]
                return [v[i, ...] for i in range(v.shape[0])]
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]

        def _clear():
            dpg.set_value(self._track_line_xy_tag, [[], []])
            dpg.set_value(self._track_line_xt_tag, [[], []])
            dpg.set_value(self._track_line_yt_tag, [[], []])

        if self.last_result is None:
            _clear()
            return
        if not self._cluster_sorted_idx:
            _clear()
            return
    
        current_cluster_idx = self._cluster_sorted_idx[self._cluster_pos]
        is_test_overlay = (self._last_test_cluster_idx is not None)
    
        if is_test_overlay and current_cluster_idx != self._last_test_cluster_idx:
            _clear()
            return
    
        out = self.last_result
    
        def _get(name: str):
            v = getattr(out, name, None)
            if v is None and isinstance(out, dict):
                v = out.get(name)
            return v
    
        x_plot = _get("x_plot")
        y_plot = _get("y_plot")
        t_plot = _get("t_plot")
    
        if x_plot is None or y_plot is None or t_plot is None:
            _clear()
            return

        x_plot = _to_track_list(x_plot)
        y_plot = _to_track_list(y_plot)
        t_plot = _to_track_list(t_plot)
    
        n_tracks = min(len(x_plot), len(y_plot), len(t_plot))
        if n_tracks == 0:
            _clear()
            return
            
        if is_test_overlay:
            k = 0
        else:
            k = self._cluster_idx_to_track_idx(out, int(current_cluster_idx))
            if k is None:
                _clear()
                return
    
        if k < 0 or k >= n_tracks:
            _clear()
            return
    
        x = np.asarray(x_plot[k], dtype=float).ravel()
        y = np.asarray(y_plot[k], dtype=float).ravel()
        t = np.asarray(t_plot[k], dtype=float).ravel()

        n = min(len(x), len(y), len(t))
        if n == 0:
            _clear()
            return
        x = x[:n]
        y = y[:n]
        t = t[:n]

        m_xy = (~np.isnan(x)) & (~np.isnan(y))
        x_xy = x[m_xy]
        y_xy = y[m_xy]
    
        m_xt = (~np.isnan(t)) & (~np.isnan(x))
        t_xt = t[m_xt] * 1e-6  
        x_xt = x[m_xt]
    
        m_yt = (~np.isnan(t)) & (~np.isnan(y))
        t_yt = t[m_yt] * 1e-6
        y_yt = y[m_yt]
    
        dpg.set_value(self._track_line_xy_tag, [x_xy.tolist(), y_xy.tolist()])
        dpg.set_value(self._track_line_xt_tag, [t_xt.tolist(), x_xt.tolist()])
        dpg.set_value(self._track_line_yt_tag, [t_yt.tolist(), y_yt.tolist()])

    def _on_prev_cluster(self, sender=None, app_data=None, user_data=None) -> None:
        if not self._cluster_sorted_idx:
            self._refresh_cluster_order_if_needed(force=True)
        if not self._cluster_sorted_idx:
            return
        self._cluster_pos = (self._cluster_pos - 1) % len(self._cluster_sorted_idx)
        self._update_cluster_plot()
        self._update_tracking_overlay()

    def _on_next_cluster(self, sender=None, app_data=None, user_data=None) -> None:
        if not self._cluster_sorted_idx:
            self._refresh_cluster_order_if_needed(force=True)
        if not self._cluster_sorted_idx:
            return
        self._cluster_pos = (self._cluster_pos + 1) % len(self._cluster_sorted_idx)
        self._update_cluster_plot()
        self._update_tracking_overlay()

    # ------------------------------------------------------------------
    # Background processing (mirrors DetectionTab)
    # ------------------------------------------------------------------
    def process_background_tasks(self) -> None:
        while not self._log_queue.empty():
            line = self._log_queue.get_nowait()
            if line in ("__DONE__", "__DONE_TEST__"):
                if dpg.does_item_exist(self._run_status_tag):
                    dpg.set_value(self._run_status_tag, "Tracking done.")
                if dpg.does_item_exist(self._run_button_tag):
                    dpg.configure_item(self._run_button_tag, enabled=True)
                if dpg.does_item_exist("track_test_button"):
                    dpg.configure_item("track_test_button", enabled=True)
                if dpg.does_item_exist("track_stop_button"):
                    dpg.configure_item("track_stop_button", enabled=False)
            
                self._update_tracking_overlay()

                if line == "__DONE__":
                    self._update_tracking_stats()
                    if float(self.params.get("track_vmax", 0.0)) == 0.0:
                        vmax_suggest = self._suggest_vmax_from_results()
                        if vmax_suggest > 0 and dpg.does_item_exist(self._pf_vmax_input_tag):
                            dpg.set_value(self._pf_vmax_input_tag, vmax_suggest)
                            self.params["track_vmax"] = vmax_suggest
                    self._update_track_duration_histogram()
                    self._update_tracking_pseudoframe_overlay()
            
                self._status_log_append("Tracking finished.")
                continue
            self._status_log_append(line)

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------
    def _on_run_tracking_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        self._stop_event.clear()
        if self._run_thread is not None and self._run_thread.is_alive():
            self._status_log_append("Tracking is already running.")
            return
        
        if dpg.does_item_exist("track_test_button"):
            dpg.configure_item("track_test_button", enabled=False)
        dpg.configure_item(self._run_button_tag, enabled=False)
        dpg.set_value(self._run_status_tag, "Running tracking...")
        
        self._last_test_cluster_idx = None
        for tag in (self._track_line_xy_tag, self._track_line_xt_tag, self._track_line_yt_tag):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, [[], []])

        try:
            ctx = self._build_engine_context()
            self._last_run_ctx = ctx
        except Exception as e:
            dpg.set_value(self._run_status_tag, f"Input error: {e}")
            dpg.configure_item(self._run_button_tag, enabled=True)
            return

        def worker():
            stdout_writer = QueueWriter(self._log_queue)
            stderr_writer = QueueWriter(self._log_queue)
            try:
                with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
                    out = self.engine.run(ctx)
                    self.last_result = out
                self._log_queue.put("[TRACKING] Done.")
            except TrackingCancelled:
                self._log_queue.put("[TRACKING] Run cancelled by user.")
            except Exception as e:
                self._log_queue.put(f"[TRACKING] ERROR: {e}")
            finally:
                self._log_queue.put("__DONE__")

        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()

    def _on_stop_tracking_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        self._stop_event.set()
        self._status_log_append("Stop requested... (tracking.py must check stop_event to interrupt)")

    # ------------------------------------------------------------------
    # Engine context (analog zu DetectionTab)
    # ------------------------------------------------------------------
    def _build_engine_context(self) -> dict[str, Any]:
        if self.preprocessing_tab is None:
            raise RuntimeError("PreprocessingTab ist nicht verbunden – keine Daten verfügbar.")
        if self.detection_tab is None:
            raise RuntimeError("DetectionTab ist nicht verbunden – keine Detection-Ergebnisse verfügbar.")
        if getattr(self.detection_tab, "last_result", None) is None:
            raise RuntimeError("Keine Detection-Ergebnisse gefunden. Bitte zuerst Detection ausführen.")

        det_out = self.detection_tab.last_result

        det_method = self.detection_tab.params.get("detection_method", "pixelwise extension")
        pseudo_images = (det_method == "pseudo-frame")
        lightweight_mode = bool(self.detection_tab.params.get("lightweight_mode", False)) or pseudo_images
        
        if lightweight_mode or pseudo_images:
            n_tracks_data = len(det_out.t_m)
            max_len_data = max((len(a) for a in det_out.t_m), default=0)
        else:
            n_tracks_data = len(det_out.x_clust)
            max_len_data = max((len(a) for a in det_out.t_clust), default=0)

        user_max_tracks = int(self.params.get("max_tracks", n_tracks_data))
        user_max_length = int(self.params.get("max_length", max_len_data))
        
        max_tracks = min(user_max_tracks, n_tracks_data) if n_tracks_data > 0 else user_max_tracks
        max_length = min(user_max_length, max_len_data) if max_len_data > 0 else user_max_length

        max_tracks = max(1, max_tracks)
        max_length = max(1, max_length)

        method = self.params.get("tracking_method", self._track_method_options[0])
        
        Kalman_afterwards = False
        spline_fitting = False
        use_both = False
        both2 = False
        use_hybrid = False

        if method == "Kalman filtering":
            Kalman_afterwards = True
        elif method == "B-spline":
            spline_fitting = True
        elif method == "hybrid(xk,us)":
            use_both = True
        elif method == "hybrid(xk,uks)":
            both2 = True
        elif method == "hybrid(xks,uks)":
            use_hybrid = True

        correction = bool(self.params.get("correction", False))

        C = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        P = float(self.params.get("P", 1e8)) * np.eye(3, dtype=np.float64)
        Q = np.array([
            [dpg.get_value(self.q_inputs[i][j]) for j in range(3)]
            for i in range(3)
        ], dtype=float)
        
        if Q.shape != (3, 3):
            Q = np.eye(3) * 1e-3
        R = float(self.params.get("R", 1.0))
        
        pt = self.preprocessing_tab
        dt = pt.params["accumulation_time_ms"]*1000

        ctx: dict[str, Any] = {
            "lightweight_mode": lightweight_mode,
            "pseudo_images": pseudo_images,

            "Kalman_afterwards": Kalman_afterwards,
            "use_both": use_both,
            "both2": both2,
            "correction": correction,
            "spline_fitting": spline_fitting,
            "use_hybrid": use_hybrid,

            "max_tracks": max_tracks,
            "max_length": max_length,
            "num_worker": int(self.params.get("num_worker", 8)),
            "dt": dt,
            "resol": float(self.params.get("resol", 1.0)),

            "P": P,
            "C": C,
            "Q": Q,
            "R": R,

            "stop_event": self._stop_event,
        }

        if method == "B-spline" or method.startswith("hybrid"):
            ctx["target_RMS"] = float(dpg.get_value(self.target_rms))
        if dpg.get_value(self.apply_correction):
            ctx["correction"] = True
            ctx["beta"] = float(dpg.get_value(self.beta))
        else:
            ctx["correction"] = False

        if lightweight_mode or pseudo_images:
            ctx["t_m"] = det_out.t_m
            ctx["x_m"] = det_out.x_m
            ctx["y_m"] = det_out.y_m
            
            ctx["t_clust"] = det_out.t_clust
            ctx["x_clust"] = det_out.x_clust
            ctx["y_clust"] = det_out.y_clust
        else:
            ctx["t_clust"] = det_out.t_clust
            ctx["x_clust"] = det_out.x_clust
            ctx["y_clust"] = det_out.y_clust

        return ctx

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _status_log_append(self, line: str) -> None:
        if not dpg.does_item_exist("track_status_log_text"):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {line}"
        current = dpg.get_value("track_status_log_text") or ""
        dpg.set_value("track_status_log_text", (line + ("\n" if current else "") + current))

    def _on_track_method_change(self, sender=None, app_data=None, user_data=None):
        method = app_data
        self.params["tracking_method"] = method
    
        is_kalman = (method == "Kalman filtering")
        is_bspline = (method == "B-spline")
        is_hybrid = method.startswith("hybrid")

        dpg.configure_item("track_group_kalman_params", show=(is_kalman or is_hybrid))
    
        dpg.configure_item("track_group_bspline_params", show=(is_bspline or is_hybrid))

    def _set_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def _on_apply_correction_change(self, sender, app_data, user_data=None):
        dpg.configure_item("track_group_beta", show=bool(app_data))
        
    def _on_test_tracking_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        """Run tracking only on the currently shown cluster (plot index)."""
        self._stop_event.clear()
        if dpg.does_item_exist("track_stop_button"):
            dpg.configure_item("track_stop_button", enabled=True)
        if dpg.does_item_exist(self._run_button_tag):
            dpg.configure_item(self._run_button_tag, enabled=False)
        if dpg.does_item_exist("track_test_button"):
            dpg.configure_item("track_test_button", enabled=False)
        if self._run_thread is not None and self._run_thread.is_alive():
            self._status_log_append("Tracking is already running.")
            return
        if dpg.does_item_exist(self._run_status_tag):
            dpg.set_value(self._run_status_tag, "Running TEST tracking (current cluster)...")

        self._refresh_cluster_order_if_needed(force=False)
        if not self._cluster_sorted_idx:
            self._status_log_append("No cluster available for test tracking. Run detection first.")
            dpg.configure_item(self._run_button_tag, enabled=True)
            dpg.configure_item("track_test_button", enabled=True)
            if dpg.does_item_exist("track_stop_button"):
                dpg.configure_item("track_stop_button", enabled=False)
            dpg.set_value(self._run_status_tag, "No cluster available.")
            return
    
        cluster_idx = self._cluster_sorted_idx[self._cluster_pos]
        self._last_test_cluster_idx = cluster_idx
    
        try:
            ctx = self._build_engine_context()
            ctx = self._slice_ctx_to_single_cluster(ctx, cluster_idx)
            self._last_run_ctx = ctx
        except Exception as e:
            dpg.set_value(self._run_status_tag, f"Input error: {e}")
            dpg.configure_item(self._run_button_tag, enabled=True)
            dpg.configure_item("track_test_button", enabled=True)
            if dpg.does_item_exist("track_stop_button"):
                dpg.configure_item("track_stop_button", enabled=False)
            return
    
        def worker():
            stdout_writer = QueueWriter(self._log_queue)
            stderr_writer = QueueWriter(self._log_queue)
            try:
                with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
                    out = self.engine.run(ctx)
                    self.last_result = out
                self._log_queue.put(f"[TRACKING] Test run done (cluster {cluster_idx}).")
            except TrackingCancelled:
                self._log_queue.put("[TRACKING] Test run cancelled by user.")
            except Exception as e:
                self._log_queue.put(f"[TRACKING] TEST ERROR: {e}")
            finally:
                self._log_queue.put("__DONE_TEST__")
    
        self._run_thread = threading.Thread(target=worker, daemon=True)
        self._run_thread.start()
        
        
    def _slice_ctx_to_single_cluster(self, ctx: dict[str, Any], idx: int) -> dict[str, Any]:
        ctx = dict(ctx)  

        if ctx.get("pseudo_images", False) or ctx.get("lightweight_mode", False):
            for key in ("t_m", "x_m", "y_m"):
                if key in ctx:
                    src = ctx[key]
                    ctx[key] = [src[idx]]
        else:
            for key in ("t_clust", "x_clust", "y_clust"):
                if key in ctx:
                    src = ctx[key]
                    ctx[key] = [src[idx]]
        ctx["max_tracks"] = 1
    
        return ctx
    
    def _on_clear_tracking_results_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        self.last_result = None
        self._last_test_cluster_idx = None

        for tag in (self._track_line_xy_tag, self._track_line_xt_tag, self._track_line_yt_tag):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, [[], []])

        self._status_log_append("Tracking results cleared.")
        if dpg.does_item_exist(self._run_status_tag):
            dpg.set_value(self._run_status_tag, "Tracking results cleared.")
            
        if dpg.does_item_exist(self._track_run_stats_text_tag):
            dpg.set_value(self._track_run_stats_text_tag, "No results yet.")
            
        self._update_tracking_pseudoframe_overlay()
        self._update_track_duration_histogram()
            
    def _update_tracking_stats(self) -> None:
        if not dpg.does_item_exist(self._track_run_stats_text_tag):
            return
    
        if self.last_result is None:
            dpg.set_value(self._track_run_stats_text_tag, "No results yet.")
            return
    
        out = self.last_result
    
        def _get(name: str):
            v = getattr(out, name, None)
            if v is None and isinstance(out, dict):
                v = out.get(name)
            return v
    
        t_plot = _get("t_plot")
        if t_plot is None:
            dpg.set_value(self._track_run_stats_text_tag, "No tracking output (t_plot missing).")
            return

        if not isinstance(t_plot, (list, tuple)):
            t_plot = [t_plot]
    
        lengths_s = []
        for t in t_plot:
            t = np.asarray(t, dtype=float).ravel()
            t = t[~np.isnan(t)]
            if t.size < 2:
                continue
            dur_s = (float(np.max(t)) - float(np.min(t))) * 1e-6  
            if dur_s > 0:
                lengths_s.append(dur_s)
    
        n_tracks = len(lengths_s)
        mean_len = float(np.mean(lengths_s))*1e3 if n_tracks > 0 else 0.0
    
        dpg.set_value(
            self._track_run_stats_text_tag,
            f"Number of tracks: {n_tracks}\n"
            f"Mean track length [ms]: {mean_len:.6g}"
        )
        
    def _on_track_timefactor_change(self, v: float) -> None:
        try:
            v = float(v)
        except Exception:
            return
        if v <= 0:
            v = 1.0
            if dpg.does_item_exist(self._pf_timefactor_input_tag):
                dpg.set_value(self._pf_timefactor_input_tag, v)
        self.params["track_timefactor"] = v
        self._update_tracking_pseudoframe_overlay()
        
    def _on_pf_prev(self, sender=None, app_data=None, user_data=None):
        if self.preprocessing_tab is None:
            return
        self.preprocessing_tab.on_prev_pseudoframe(sender, app_data, user_data)
        self._update_tracking_pseudoframe_overlay()
    
    def _on_pf_next(self, sender=None, app_data=None, user_data=None):
        if self.preprocessing_tab is None:
            return
        self.preprocessing_tab.on_next_pseudoframe(sender, app_data, user_data)
        self._update_tracking_pseudoframe_overlay()
        
    def _update_tracking_pseudoframe_overlay(self) -> None:
        if not dpg.does_item_exist(self._pf_tracks_node_tag):
            return

        if dpg.does_item_exist(self._pf_tracks_node_tag):
            dpg.delete_item(self._pf_tracks_node_tag)
        
        with dpg.draw_node(tag=self._pf_tracks_node_tag, parent=self._pf_drawlist_tag):
            pass
    
        if self.preprocessing_tab is None or self.last_result is None:
            return
    
        out = self.last_result
    
        def _get(name: str):
            v = getattr(out, name, None)
            if v is None and isinstance(out, dict):
                v = out.get(name)
            return v
    
    
        x_plot = [np.asarray(x, dtype=float) - 1 for x in _get("x_plot")]
        y_plot = [np.asarray(y, dtype=float) - 1 for y in _get("y_plot")]
        t_plot = [np.asarray(t, dtype=float)       for t in _get("t_plot")]
        x_plotv = [np.asarray(t, dtype=float)       for t in _get("x_plotv")]
        y_plotv = [np.asarray(t, dtype=float)       for t in _get("y_plotv")]
    
        if x_plot is None or y_plot is None or t_plot is None or x_plotv is None or y_plotv is None:
            return

        if not isinstance(x_plot, (list, tuple)): x_plot = [x_plot]
        if not isinstance(y_plot, (list, tuple)): y_plot = [y_plot]
        if not isinstance(t_plot, (list, tuple)): t_plot = [t_plot]
        if not isinstance(x_plotv, (list, tuple)): x_plotv = [x_plotv]
        if not isinstance(y_plotv, (list, tuple)): y_plotv = [y_plotv]
    
        n_tracks = min(len(x_plot), len(y_plot), len(t_plot), len(x_plotv), len(y_plotv))
        if n_tracks == 0:
            return

        pt = self.preprocessing_tab
        events = pt.filtered_events if pt.filtered_events is not None else pt.raw_events
        if events is None:
            return
        _, _, T_all, _ = events
    
        acc_ms = float(getattr(pt, "params", {}).get("accumulation_time_ms", 2.0))
        acc_us = acc_ms * 1000.0
        
        factor = float(self.params.get("track_timefactor", 2.0))
        if factor <= 1.0:
            factor = 1.0
    
        t0 = pt.display_t0
        if t0 is None:
            t0 = float(np.min(T_all)) if len(T_all) else 0.0
    
    
        T_min = float(t0) - (factor-1.0) * acc_us
        T_max = float(t0) + acc_us
        vmins = []
        vmaxs = []
        for i in range(n_tracks):
            t = np.asarray(t_plot[i], dtype=float).ravel()
            x = np.asarray(x_plot[i], dtype=float).ravel()
            y = np.asarray(y_plot[i], dtype=float).ravel()
            xv = np.asarray(x_plotv[i], dtype=float).ravel()
            yv = np.asarray(y_plotv[i], dtype=float).ravel()
    
            n = min(len(t), len(x), len(y), len(xv), len(yv))
            if n < 2:
                continue
            t, x, y, xv, yv = t[:n], x[:n], y[:n], xv[:n], yv[:n]
    
            m = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(xv) & np.isfinite(yv)
            m &= (t >= T_min) & (t <= T_max)
            if np.count_nonzero(m) < 2:
                continue
    
            spd = np.sqrt(xv[m] ** 2 + yv[m] ** 2)
            if spd.size:
                vmins.append(float(np.min(spd)))
                vmaxs.append(float(np.max(spd)))
    
        if not vmins:
            return
    
        vmin = min(vmins)
        vmin = 0.0
        vmax_data = max(vmaxs)
        vmax_user = float(self.params.get("track_vmax", 0.0))
        vmax = vmax_user if vmax_user > 0.0 else vmax_data

        if vmax <= vmin:
            vmax = vmin + 1e-12
    
        cmap = cm.get_cmap("viridis")
    
        def _speed_to_color(s: float):
            a = (s - vmin) / (vmax - vmin)
            a = 0.0 if a < 0.0 else 1.0 if a > 1.0 else a
            r, g, b, _ = cmap(a)
            return (int(r * 255), int(g * 255), int(b * 255), 255)

        for i in range(n_tracks):
            t = np.asarray(t_plot[i], dtype=float).ravel()
            x = np.asarray(x_plot[i], dtype=float).ravel()
            y = np.asarray(y_plot[i], dtype=float).ravel()
            xv = np.asarray(x_plotv[i], dtype=float).ravel()
            yv = np.asarray(y_plotv[i], dtype=float).ravel()
    
            n = min(len(t), len(x), len(y), len(xv), len(yv))
            if n < 2:
                continue
            t, x, y, xv, yv = t[:n], x[:n], y[:n], xv[:n], yv[:n]
    
            m = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(xv) & np.isfinite(yv)
            m &= (t >= T_min) & (t <= T_max)
            if np.count_nonzero(m) < 2:
                continue
    
            t = t[m]; x = x[m]; y = y[m]; xv = xv[m]; yv = yv[m]
            spd = np.sqrt(xv ** 2 + yv ** 2)

            for k in range(len(x) - 1):
                col = _speed_to_color(float(spd[k]))
                dpg.draw_line(
                    p1=[float(x[k]), float(y[k])],
                    p2=[float(x[k + 1]), float(y[k + 1])],
                    color=col,
                    thickness=2.0,
                    parent=self._pf_tracks_node_tag,
                )
                
    def _on_track_vmax_change(self, v: float) -> None:
        try:
            v = float(v)
        except Exception:
            return
        if v < 0:
            v = 0.0
            if dpg.does_item_exist(self._pf_vmax_input_tag):
                dpg.set_value(self._pf_vmax_input_tag, v)
        self.params["track_vmax"] = v
        self._update_tracking_pseudoframe_overlay()
        
    def _suggest_vmax_from_results(self) -> float:
        out = self.last_result
        if out is None:
            return 0.0
    
        def _get(name: str):
            v = getattr(out, name, None)
            if v is None and isinstance(out, dict):
                v = out.get(name)
            return v
    
        xv = _get("x_plotv")
        yv = _get("y_plotv")
        if xv is None or yv is None:
            return 0.0
    
        if not isinstance(xv, (list, tuple)):
            xv = [xv]
        if not isinstance(yv, (list, tuple)):
            yv = [yv]
    
        speeds = []
        n = min(len(xv), len(yv))
        for i in range(n):
            a = np.asarray(xv[i], dtype=float).ravel()
            b = np.asarray(yv[i], dtype=float).ravel()
            m = np.isfinite(a) & np.isfinite(b)
            if np.count_nonzero(m) == 0:
                continue
            spd = np.sqrt(a[m] ** 2 + b[m] ** 2)
            if spd.size:
                speeds.append(spd)
    
        if not speeds:
            return 0.0
    
        all_spd = np.concatenate(speeds)
        med = float(np.median(all_spd)) if all_spd.size else 0.0
        return 2.5 * med
    
    def _update_track_duration_histogram(self) -> None:
        if not dpg.does_item_exist(self.duration_hist_texture_tag):
            return
    
        out = getattr(self, "last_result", None)
        durations_us = []
    
        if out is not None:
            t_plot = getattr(out, "t_plot", None)
            if t_plot is None and isinstance(out, dict):
                t_plot = out.get("t_plot")
    
            if t_plot is not None:
                if isinstance(t_plot, (list, tuple)):
                    tracks = t_plot
                else:
                    tracks = [t_plot]
    
                for t in tracks:
                    if t is None:
                        continue
                    tt = np.asarray(t, dtype=float).ravel()
                    tt = tt[np.isfinite(tt)]
                    if tt.size < 2:
                        continue
                    durations_us.append(float(np.max(tt) - np.min(tt)))
        W, H = 700, 580
        dpi = 100
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        
        fig.patch.set_facecolor((0.12, 0.12, 0.12))
        ax.set_facecolor((0.12, 0.12, 0.12))
        
        for spine in ax.spines.values():
            spine.set_color((0.7, 0.7, 0.7))
        
        ax.tick_params(colors=(0.85, 0.85, 0.85))
        ax.xaxis.label.set_color((0.9, 0.9, 0.9))
        ax.yaxis.label.set_color((0.9, 0.9, 0.9))
        ax.title.set_color((0.95, 0.95, 0.95))
        
        ax.grid(True, alpha=0.2)
    
        if len(durations_us) == 0:
            ax.text(0.5, 0.5, "No duration data.", ha="center", va="center", color=(0.9, 0.9, 0.9))
            ax.set_axis_off()
        else:
            d_ms = np.asarray(durations_us, dtype=float) / 1000.0  
            ax.hist(d_ms, bins=100, edgecolor=(0.85, 0.85, 0.85))
            ax.set_xlabel("Track duration [ms]")
            ax.set_ylabel("Count")
            ax.set_title("Track duration histogram")
    
        fig.tight_layout(pad=0.5)
        dpg.set_value(self.duration_hist_texture_tag, figure_to_rgba_flat(fig))
        plt.close(fig)
        
    def _sanitize_tracking_ctx(self, ctx: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(ctx, dict):
            return {}
        out = dict(ctx)
        if "stop_event" in out:
            out["stop_event"] = None
        return out
    
    def _build_tracking_payload_bytes(self) -> tuple[bytes, bytes]:
        if self.last_result is None:
            raise ValueError("No tracking results available (last_result is None).")
        ctx = getattr(self, "_last_run_ctx", None)
        if ctx is None:
            raise ValueError("No ctx available (_last_run_ctx is None). Run tracking first or load a file.")
    
        ctx2 = self._sanitize_tracking_ctx(ctx)
    
        result_bytes = pickle.dumps(self.last_result, protocol=pickle.HIGHEST_PROTOCOL)
        ctx_bytes = pickle.dumps(ctx2, protocol=pickle.HIGHEST_PROTOCOL)
        return result_bytes, ctx_bytes
    
    def browse_track_save_results_file(self, sender=None, app_data=None, user_data=None):
        if dpg.does_item_exist("track_save_results_file_dialog"):
            dpg.show_item("track_save_results_file_dialog")
    
    def browse_track_load_results_file(self, sender=None, app_data=None, user_data=None):
        if dpg.does_item_exist("track_load_results_file_dialog"):
            dpg.show_item("track_load_results_file_dialog")
    
    def on_track_save_results_file_selected(self, sender, app_data, user_data=None):
        path = str(app_data.get("file_path_name", "")).strip()
        if path:
            dpg.set_value("track_save_results_file_input", path)
    
    def on_track_load_results_file_selected(self, sender, app_data, user_data=None):
        path = str(app_data.get("file_path_name", "")).strip()
        if path:
            dpg.set_value("track_load_results_file_input", path)
    
    def save_tracking_results(self, sender=None, app_data=None, user_data=None):
        try:
            path = str(dpg.get_value("track_save_results_file_input") or "").strip()
            ext = str(dpg.get_value("track_save_results_format") or ".npz").strip().lower()
    
            if not path:
                self.browse_track_save_results_file()
                return
    
            p = Path(path)
            if p.suffix.lower() not in [".npz", ".mat"]:
                p = p.with_suffix(ext)
    
            result_bytes, ctx_bytes = self._build_tracking_payload_bytes()
            meta = {"format_version": 1, "saved_at": datetime.now().isoformat(timespec="seconds")}
    
            if p.suffix.lower() == ".npz":
                np.savez_compressed(
                    p,
                    result_pickle=np.frombuffer(result_bytes, dtype=np.uint8),
                    ctx_pickle=np.frombuffer(ctx_bytes, dtype=np.uint8),
                    meta=np.array([meta], dtype=object),
                )
            else:
                sio.savemat(
                    p,
                    {
                        "result_pickle": np.frombuffer(result_bytes, dtype=np.uint8),
                        "ctx_pickle": np.frombuffer(ctx_bytes, dtype=np.uint8),
                        "meta": np.array([meta], dtype=object),
                    },
                    do_compression=True,
                )
    
            dpg.set_value("track_save_results_status", f"Saved: {p}")
        except Exception as e:
            dpg.set_value("track_save_results_status", f"Save failed: {e}")
    
    def load_tracking_results(self, sender=None, app_data=None, user_data=None):
        try:
            path = str(dpg.get_value("track_load_results_file_input") or "").strip()
            if not path:
                self.browse_track_load_results_file()
                return
    
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(p)
    
            if p.suffix.lower() == ".npz":
                data = np.load(p, allow_pickle=True)
                result_bytes = bytes(np.array(data["result_pickle"], dtype=np.uint8).tobytes())
                ctx_bytes = bytes(np.array(data["ctx_pickle"], dtype=np.uint8).tobytes())
            elif p.suffix.lower() == ".mat":
                data = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
                result_bytes = bytes(np.array(data["result_pickle"], dtype=np.uint8).tobytes())
                ctx_bytes = bytes(np.array(data["ctx_pickle"], dtype=np.uint8).tobytes())
            else:
                raise ValueError("Unsupported file type. Use .npz or .mat.")
    
            self.last_result = pickle.loads(result_bytes)
            self._last_run_ctx = pickle.loads(ctx_bytes)

            ctx_params = self._last_run_ctx if isinstance(self._last_run_ctx, dict) else {}
            self.apply_loaded_track_ctx_to_gui(ctx_params)

            self._update_tracking_overlay()
            self._update_tracking_stats()
            self._update_track_duration_histogram()
            self._update_tracking_pseudoframe_overlay()
    
            if dpg.does_item_exist(self._track_run_stats_text_tag):
                dpg.set_value(self._track_run_stats_text_tag, f"Loaded results from: {p.name}")
    
            dpg.set_value("track_load_results_status", f"Loaded: {p}")
        except Exception as e:
            dpg.set_value("track_load_results_status", f"Load failed: {e}")
            
    def apply_loaded_track_ctx_to_gui(self, ctx: dict[str, Any]):
        if not isinstance(ctx, dict):
            return

        if bool(ctx.get("use_hybrid", False)):
            method = "hybrid(xks,uks)"
        elif bool(ctx.get("both2", False)):
            method = "hybrid(xk,uks)"
        elif bool(ctx.get("use_both", False)):
            method = "hybrid(xk,us)"
        elif bool(ctx.get("spline_fitting", False)):
            method = "B-spline"
        elif bool(ctx.get("Kalman_afterwards", False)):
            method = "Kalman filtering"
        else:
            method = self.params.get("tracking_method", "Kalman filtering")
    
        self.params["tracking_method"] = method
        if dpg.does_item_exist(self._track_method_combo_tag):
            dpg.set_value(self._track_method_combo_tag, method)
        self._on_track_method_change(None, method, None)

        if "num_worker" in ctx:
            try:
                v = int(ctx["num_worker"])
                self.params["num_worker"] = v
                if dpg.does_item_exist("track_input_num_worker"):
                    dpg.set_value("track_input_num_worker", v)
            except Exception:
                pass
    
        if "resol" in ctx:
            try:
                v = float(ctx["resol"])
                self.params["resol"] = v
                if dpg.does_item_exist("track_input_resol"):
                    dpg.set_value("track_input_resol", v)
            except Exception:
                pass

        if "P" in ctx:
            try:
                Pm = np.asarray(ctx["P"], dtype=float)
                p_scalar = float(np.mean(np.diag(Pm))) if Pm.ndim == 2 and Pm.shape[0] == Pm.shape[1] else float(Pm)
                self.params["P"] = p_scalar
                if dpg.does_item_exist("track_input_P"):
                    dpg.set_value("track_input_P", p_scalar)
            except Exception:
                pass
    
        if "R" in ctx:
            try:
                v = float(ctx["R"])
                self.params["R"] = v
                if dpg.does_item_exist("track_input_R"):
                    dpg.set_value("track_input_R", v)
            except Exception:
                pass
    
        if "Q" in ctx:
            try:
                Qm = np.asarray(ctx["Q"], dtype=float)
                if Qm.shape == (3, 3) and hasattr(self, "q_inputs"):
                    for i in range(3):
                        for j in range(3):
                            tag = self.q_inputs[i][j]
                            if dpg.does_item_exist(tag):
                                dpg.set_value(tag, float(Qm[i, j]))
            except Exception:
                pass

        if "correction" in ctx:
            try:
                corr = bool(ctx["correction"])
                self.params["correction"] = corr
                if hasattr(self, "apply_correction") and dpg.does_item_exist(self.apply_correction):
                    dpg.set_value(self.apply_correction, corr)
                self._on_apply_correction_change(None, corr, None)
            except Exception:
                pass
    
        if "beta" in ctx:
            try:
                v = float(ctx["beta"])
                self.params["beta"] = v
                if hasattr(self, "beta") and dpg.does_item_exist(self.beta):
                    dpg.set_value(self.beta, v)
            except Exception:
                pass

        if "target_RMS" in ctx:
            try:
                v = float(ctx["target_RMS"])
                self.params["target_RMS"] = v
                if hasattr(self, "target_rms") and dpg.does_item_exist(self.target_rms):
                    dpg.set_value(self.target_rms, v)
            except Exception:
                pass

        if "track_timefactor" in ctx and dpg.does_item_exist(self._pf_timefactor_input_tag):
            try:
                v = float(ctx["track_timefactor"])
                self.params["track_timefactor"] = v
                dpg.set_value(self._pf_timefactor_input_tag, v)
            except Exception:
                pass
    
        if "track_vmax" in ctx and dpg.does_item_exist(self._pf_vmax_input_tag):
            try:
                v = float(ctx["track_vmax"])
                self.params["track_vmax"] = v
                dpg.set_value(self._pf_vmax_input_tag, v)
            except Exception:
                pass
    
    def _cluster_idx_to_track_idx(self, out, cluster_idx: int) -> int | None:
        m = getattr(out, "track_to_cluster", None)
        if m is None and isinstance(out, dict):
            m = out.get("track_to_cluster")
    
        if m is None:
            return int(cluster_idx)
    
        m = np.asarray(m, dtype=int).ravel()
        for track_idx, c in enumerate(m):
            if int(c) == int(cluster_idx):
                return int(track_idx)
        return None