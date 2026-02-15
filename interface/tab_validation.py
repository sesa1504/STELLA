from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, TYPE_CHECKING

import dearpygui.dearpygui as dpg
import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap
from datetime import datetime
import sys
from scipy.spatial import cKDTree
import pickle
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import DEFAULT_PARAMS, PSEUDOFAME_DIMENSIONS

if TYPE_CHECKING:
    from interface.tab_preprocessing import PreprocessingTab
    
PSEUDOFRAME_CMAP = ListedColormap(
    np.array([
        [30,  37,  52],
        [64, 124, 198],
        [220, 226, 238],
    ], dtype=np.float32) / 255.0
)

class ValidationTab:
    label: str = "Validation"
    
    def __init__(
        self,
        preprocessing_tab: Optional[PreprocessingTab] = None,
        tracking_tab: Any = None,
    ) -> None:
        self.preprocessing_tab = preprocessing_tab
        self.tracking_tab = tracking_tab

        self.params: dict[str, Any] = deepcopy(DEFAULT_PARAMS)

        self.params.setdefault("apply_track_quality_filter", False)
        self.params.setdefault("track_quality_threshold", 2.0)
        self.params.setdefault("track_quality_quantile", 0.95)
        self.params.setdefault("track_quality_w_angle", 1.0)
        self.params.setdefault("track_quality_w_velo", 1.0)
        self.params.setdefault("track_quality_w_path", 1.0)

        self.params.setdefault("track_timefactor", 2.0)
        self.pseudoframe_dimensions = PSEUDOFAME_DIMENSIONS  
        
        self._pf_status_tag = "vali_pseudoframe_status"
        self._pf_time_info_tag = "vali_pseudoframe_time_info"
        self._pf_drawlist_tag = "vali_pf_drawlist"
        self._pf_drawimage_tag = "vali_pf_drawimage"
        self._pf_tracks_node_tag = "vali_pf_tracks_node"
        
        self._pf_timefactor_input_tag = "vali_pf_timefactor"
        
        self._run_button_tag = "vali_run_button"
        self._run_status_tag = "vali_run_status"

        self.flag: Optional[np.ndarray] = None
        self.score: Optional[np.ndarray] = None
        self.flag_points = None     
        self.flag_track = None     

        self.last_score: Optional[np.ndarray] = None
        self.last_flag: Optional[np.ndarray] = None
        self._last_valid_idx: Optional[np.ndarray] = None
        
        self.params.setdefault("apply_neighborhood_filter", False)
        self.params.setdefault("nb_dt_factor", 2.0)        
        self.params.setdefault("nb_radius", 30.0)        
        self.params.setdefault("nb_mad_factor", 30.0)
        self.params.setdefault("nb_angle_factor", 20.0)
        self.params.setdefault("nb_min_neighbors", 3)
        self.params.setdefault("nb_min_track_points", 3)   

        self.point_keep = None
        self.comp_dev = None
        self.dir_dev = None
        self.comp_flag = None
        self.dir_flag = None
        
    def build(self, parent: int) -> None:
        with dpg.group(parent=parent):
            with dpg.group(horizontal=True):
                with dpg.child_window(label="Validation", width=500, height=950):
                    self._build_data_panel()

                with dpg.child_window(label="Visualization", width=1920 - 550, height=950):
                    dpg.add_text("VISUALIZATION", color=(100, 200, 255))
                    dpg.add_separator()
                    self._build_validation_pseudoframe_panel()
                
        self._status_log_append("Ready.")
        
    def _build_data_panel(self) -> None:
        with dpg.theme() as info_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 150, 200))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 110, 160))
        
                dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 130, 180, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10) 
        dpg.add_text("VALIDATION", color=(100, 200, 255))
        dpg.add_separator()

        dpg.add_text("Track quality filter", color=(200, 200, 100))

        dpg.add_checkbox(
            label="apply track quality filter",
            default_value=bool(self.params.get("apply_track_quality_filter", False)),
            callback=lambda s, a: self._on_param_changed("apply_track_quality_filter", a),
        )

        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="score threshold",
                default_value=float(self.params.get("track_quality_threshold", 1.0)),
                min_value=0.0,
                min_clamped=True,
                width=140,
                callback=lambda s, a: self._on_param_changed("track_quality_threshold", a),
            )
            dpg.add_button(label="?", width=22, height=22, tag="score_info_btn")
            dpg.bind_item_theme("score_info_btn", info_button_theme)
        
            with dpg.popup("score_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "Tracks are rejected by this threshold in the score,\n"
                    "which is calculated as weighted sum of criteria\n"
                    "evaluating variations in direction, velocity and \n"
                    "path length.\n"
                    "The lower the threshold, the stricter the outlier\n"
                    "filter."
                )

        with dpg.group(horizontal=True):
            dpg.add_text("Weights", color=(160, 160, 160))
            dpg.add_button(label="?", width=22, height=22, tag="weight_info_btn")
            dpg.bind_item_theme("weight_info_btn", info_button_theme)
        
            with dpg.popup("weight_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "Weights applied to calculate the score.\n\n"
                    "directional variation: filter out strong oscillations\n"
                    "velocity variation: filter out strong velocity fluctuations\n"
                    "path length: filter out short tracks"
                )
        dpg.add_input_float(
            label="directional variation",
            default_value=float(self.params.get("track_quality_w_angle", 1.0)),
            min_value=0.0,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("track_quality_w_angle", a),
        )
        dpg.add_input_float(
            label="velocity variation",
            default_value=float(self.params.get("track_quality_w_velo", 1.0)),
            min_value=0.0,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("track_quality_w_velo", a),
        )
        dpg.add_input_float(
            label="path length",
            default_value=float(self.params.get("track_quality_w_path", 1.0)),
            min_value=0.0,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("track_quality_w_path", a),
        )
        dpg.add_separator()
        
        dpg.add_text("Neighborhood filter", color=(200, 200, 100))

        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label="apply neighborhood filter",
                default_value=bool(self.params.get("apply_neighborhood_filter", False)),
                callback=lambda s, a: self._on_param_changed("apply_neighborhood_filter", a),
            )
            dpg.add_button(label="?", width=22, height=22, tag="neighbor_info_btn")
            dpg.bind_item_theme("neighbor_info_btn", info_button_theme)
        
            with dpg.popup("neighbor_info_btn", mousebutton=dpg.mvMouseButton_Left):
                dpg.add_text(
                    "Outlier detection based on the neighboring tracks.\n"
                    "Parts of a track are rejected, if the velocity or\n"
                    "direction varies to a greater extend than the set\n"
                    "thresholds (in units of standard deviation) from \n"
                    "the mean values of neighboring tracks."
                )
        
        dpg.add_input_float(
            label="time window (× accumulation time)",
            default_value=float(self.params.get("nb_dt_factor", 2.0)),
            min_value=0.1,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("nb_dt_factor", a),
        )
        
        dpg.add_input_float(
            label="neigborhood radius",
            default_value=float(self.params.get("nb_radius", 30.0)),
            min_value=0.1,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("nb_radius", a),
        )
        
        dpg.add_input_float(
            label="allowed velocity variation",
            default_value=float(self.params.get("nb_mad_factor", 10.0)),
            min_value=0.1,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("nb_mad_factor", a),
        )
        
        dpg.add_input_float(
            label="allowed directional variation",
            default_value=float(self.params.get("nb_angle_factor", 8.0)),
            min_value=0.1,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("nb_angle_factor", a),
        )
        
        dpg.add_input_int(
            label="minimum number of neighbors",
            default_value=int(self.params.get("nb_min_neighbors", 3)),
            min_value=1,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("nb_min_neighbors", a),
        )
        
        dpg.add_input_int(
            label="minimum points per track",
            default_value=int(self.params.get("nb_min_track_points", 3)),
            min_value=1,
            min_clamped=True,
            width=140,
            callback=lambda s, a: self._on_param_changed("nb_min_track_points", a),
        )
        
        dpg.add_separator()

        dpg.add_button(label="Run validation", tag=self._run_button_tag, callback=self._on_run_validation, width=400)
        dpg.add_button(label="Clear validation", callback=self._clear_validation_clicked, width=200)

        dpg.add_text("Validation stats:", color=(200, 200, 100))
        with dpg.child_window(
            tag="validation_run_stats_box",
            height=110,
            autosize_x=True,
            border=True,
        ):
            dpg.add_text("-", tag=self._run_status_tag, wrap=480)
            
        dpg.add_separator()
        dpg.add_text("EXPORT RESULTS", color=(100, 200, 255))
        dpg.add_text("Save validated tracks:", color=(200, 200, 100))
        
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="vali_save_results_file_input", width=300, hint="Save as .npz or .mat")
            dpg.add_button(label="Browse", callback=self.browse_vali_save_results_file, width=80)
            dpg.add_combo(items=[".npz", ".mat"], default_value=".npz", tag="vali_save_results_format", width=70)
        
        dpg.add_button(label="Save results", callback=self.save_validation_results, width=120)
        dpg.add_text("", tag="vali_save_results_status", wrap=460, color=(100, 255, 100))

        if not dpg.does_item_exist("vali_save_results_file_dialog"):
            with dpg.file_dialog(
                tag="vali_save_results_file_dialog",
                label="Save validated tracks",
                directory_selector=False,
                show=False,
                callback=self.on_vali_save_results_file_selected,
                cancel_callback=lambda s, a: dpg.set_value("vali_save_results_status", "Save cancelled"),
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))
        
    def _build_validation_pseudoframe_panel(self) -> None:
        pseudo_texture = (
            self.preprocessing_tab.pseudoframe_texture_tag
            if self.preprocessing_tab is not None
            else "preprocessing_pseudoframe_texture"
        )

        w, h = self.pseudoframe_dimensions 

        with dpg.drawlist(width=w, height=h, tag=self._pf_drawlist_tag):
            dpg.draw_image(
                pseudo_texture,
                pmin=[0, 0],
                pmax=[w, h],
                uv_min=[0.0, 0.0],
                uv_max=[1.0, 1.0],
                tag=self._pf_drawimage_tag,
            )
            with dpg.draw_node(tag=self._pf_tracks_node_tag):
                pass

        dpg.add_text("Load data in Preprocessing to generate pseudo-frame.", tag=self._pf_status_tag, wrap=1200)

        with dpg.group(horizontal=True):
            dpg.add_button(label="<< Prev", width=120, callback=self._on_pf_prev)
            dpg.add_button(label="Next >>", width=120, callback=self._on_pf_next)
            dpg.add_text("", tag=self._pf_time_info_tag, wrap=900)

        if self.preprocessing_tab is not None:
            self.preprocessing_tab.register_pseudoframe_mirror(
                status_tag=self._pf_status_tag,
                time_info_tag=self._pf_time_info_tag,
            )

        dpg.add_input_float(
            label="Track window factor (× accumulation time)",
            default_value=float(self.params.get("track_timefactor", 2.0)),
            width=240,
            format="%.2f",
            on_enter=True,
            tag=self._pf_timefactor_input_tag,
            callback=lambda s, a, u: self._on_track_timefactor_change(float(a)),
        )
        
        dpg.add_separator()
        
        dpg.add_text("Status Log", color=(200, 200, 100))
        with dpg.child_window(label="Status Log", height=90, width=720):
            dpg.add_input_text(
                tag="vali_status_log_text",
                multiline=True,
                readonly=True,
                width=695,
                height=75,
            )
        
    def _status_log_append(self, line: str) -> None:
        if not dpg.does_item_exist("vali_status_log_text"):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {line}"
        current = dpg.get_value("vali_status_log_text") or ""
        dpg.set_value("vali_status_log_text", (line + ("\n" if current else "") + current))

    # ------------------------------------------------------------------
    # Callbacks / logic
    # ------------------------------------------------------------------
    def _on_param_changed(self, key: str, value: Any) -> None:
        self.params[key] = value

    def _get_latest_tracking_outputs(self):
        tab = getattr(self, "tracking_tab", None)
        if tab is None:
            return None
        return getattr(tab, "last_result", None)

    def _clear_validation_clicked(self, sender=None, app_data=None, user_data=None) -> None:
        self.last_score = None
        self.last_flag = None
        self._last_valid_idx = None

        if dpg.does_item_exist(self._run_status_tag):
            dpg.set_value(self._run_status_tag, "-")
        if dpg.does_item_exist("vali_summary_text"):
            dpg.set_value("vali_summary_text", "")

        self.score = None
        self.flag = None

        self.flag_points = None
        self.flag_track = None
        self.point_keep = None
        self.comp_dev = None
        self.dir_dev = None
        self.comp_flag = None
        self.dir_flag = None

        for attr in [
            "validated_x_plot", "validated_y_plot", "validated_t_plot",
            "validated_x_plotv", "validated_y_plotv"
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

        if dpg.does_item_exist(self._pf_tracks_node_tag):
            try:
                dpg.delete_item(self._pf_tracks_node_tag)
            except Exception:
                pass
            with dpg.draw_node(tag=self._pf_tracks_node_tag, parent=self._pf_drawlist_tag):
                pass
    
        self._status_log_append("Cleared validation results.")
        
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
        self._update_validation_overlay()

    def _on_pf_prev(self, sender=None, app_data=None, user_data=None):
        if self.preprocessing_tab is None:
            return
        self.preprocessing_tab.on_prev_pseudoframe(sender, app_data, user_data)
        self._update_validation_overlay()

    def _on_pf_next(self, sender=None, app_data=None, user_data=None):
        if self.preprocessing_tab is None:
            return
        self.preprocessing_tab.on_next_pseudoframe(sender, app_data, user_data)
        self._update_validation_overlay()

    def _on_run_validation(self, sender=None, app_data=None, user_data=None) -> None:
        if self.tracking_tab is None or getattr(self.tracking_tab, "last_result", None) is None:
            self._set_run_status("No tracking results available. Run Tracking first.")
            self._status_log_append("Run validation failed: no tracking results.")
            return
    
        out = self.tracking_tab.last_result
    
        def _get(name: str):
            v = getattr(out, name, None)
            if v is None and isinstance(out, dict):
                v = out.get(name)
            return v
    
        x_plot = _get("x_plot")
        y_plot = _get("y_plot")
        x_plotv = _get("x_plotv")
        y_plotv = _get("y_plotv")
        t_plot = _get("t_plot")
    
        if any(v is None for v in (x_plot, y_plot, x_plotv, y_plotv, t_plot)):
            self._set_run_status("Tracking output incomplete (missing x/y/v/t).")
            self._status_log_append("Run validation failed: incomplete tracking output.")
            return

        if not isinstance(x_plot, (list, tuple)): x_plot = [x_plot]
        if not isinstance(y_plot, (list, tuple)): y_plot = [y_plot]
        if not isinstance(x_plotv, (list, tuple)): x_plotv = [x_plotv]
        if not isinstance(y_plotv, (list, tuple)): y_plotv = [y_plotv]
        if not isinstance(t_plot, (list, tuple)): t_plot = [t_plot]
    
        n_tracks = min(len(x_plot), len(y_plot), len(x_plotv), len(y_plotv), len(t_plot))
        if n_tracks == 0:
            self._set_run_status("No tracks found.")
            self._status_log_append("Run validation: no tracks.")
            return

        flag_points = [np.ones(len(t_plot[i]), dtype=bool) for i in range(n_tracks)]
    
        apply_quality = bool(self.params.get("apply_track_quality_filter", False))
        apply_nb = bool(self.params.get("apply_neighborhood_filter", False))
    
        self.score = np.zeros(n_tracks, dtype=float)
        flag_quality = np.ones(n_tracks, dtype=bool)
        
        self._status_log_append("Validation started ...")
    
        if apply_quality:
            angle_mean = np.zeros(n_tracks, dtype=float)
            angle = [[] for _ in range(n_tracks)]
            angle_d = [[] for _ in range(n_tracks)]
    
            for i in range(n_tracks):
                xv = np.asarray(x_plotv[i], dtype=float).ravel()
                yv = np.asarray(y_plotv[i], dtype=float).ravel()
                if xv.size == 0 or yv.size == 0:
                    continue
    
                angle[i] = np.arctan2(yv, xv)
                if np.ndim(angle[i]) == 0:
                    angle_d[i] = np.array([])
                else:
                    angle[i] = np.unwrap(angle[i])
                    angle_d[i] = np.diff(angle[i])
    
                if np.asarray(angle_d[i]).size > 0:
                    angle_mean[i] = np.nanmean(np.abs(angle_d[i])) * 180.0 / np.pi
    
            velo_var = np.zeros(n_tracks, dtype=float)
            for i in range(n_tracks):
                xv = np.asarray(x_plotv[i], dtype=float).ravel()
                yv = np.asarray(y_plotv[i], dtype=float).ravel()
                if xv.size == 0 or yv.size == 0:
                    continue
                velo = np.sqrt(xv**2 + yv**2)
                velo_var[i] = np.nanstd(velo) / (np.nanmean(velo) + 1e-8)
    
            def path_length(x, y):
                x = np.asarray(x, dtype=float).ravel()
                y = np.asarray(y, dtype=float).ravel()
                mask = ~np.isnan(x) & ~np.isnan(y)
                x = x[mask]
                y = y[mask]
                if x.size < 2:
                    return np.nan
                dx = np.diff(x)
                dy = np.diff(y)
                return np.hypot(dx, dy).sum()
    
            pathlength_all = np.array([path_length(x_plot[i], y_plot[i]) for i in range(n_tracks)], dtype=float)
            path_ref = np.nanmedian(pathlength_all)
            pathlength_rel = pathlength_all / (path_ref + 1e-8)
            pathlength_rel = 1.0 / (pathlength_rel + 1e-8)
    
            def robust_z(x):
                x = np.asarray(x, dtype=float)
                med = np.nanmedian(x)
                mad = np.nanmedian(np.abs(x - med))
                return (x - med) / (mad + 1e-8)
    
            angle_mean = robust_z(angle_mean)
            pathlength_rel = robust_z(pathlength_rel)
            velo_var = robust_z(velo_var)
    
            angle_mean = np.maximum(angle_mean, 0)
            pathlength_rel = np.maximum(pathlength_rel, 0)
            velo_var = np.maximum(velo_var, 0)
    
            def quantile_norm(x, q=0.95):
                ref = np.nanpercentile(x, q * 100.0)
                return x / (ref + 1e-8)
    
            q = float(self.params.get("track_quality_quantile", 0.95))
            angle_mean = quantile_norm(angle_mean, q)
            velo_var = quantile_norm(velo_var, q)
            pathlength_rel = quantile_norm(pathlength_rel, q)
    
            wa = float(self.params.get("track_quality_w_angle", 1.0))
            wv = float(self.params.get("track_quality_w_velo", 1.0))
            wp = float(self.params.get("track_quality_w_path", 1.0))
    
            score = wa * angle_mean + wv * velo_var + wp * pathlength_rel
            self.score = score
    
            thr = float(self.params.get("track_quality_threshold", 1.0))
            flag_quality = np.isfinite(score) & (score <= thr)

            for i in range(n_tracks):
                if not bool(flag_quality[i]):
                    flag_points[i][:] = False
    
            self._status_log_append(f"Track quality filter applied (thr={thr:g}).")
        else:
            self._status_log_append("Track quality filter disabled.")
    
        if apply_nb:
            acc_ms = float(getattr(self.preprocessing_tab, "params", {}).get("accumulation_time_ms", 2.0)) if self.preprocessing_tab else 2.0
            acc_us = acc_ms * 1000.0
            dt = float(self.params.get("nb_dt_factor", 2.0)) * acc_us
    
            radius = float(self.params.get("nb_radius", 30.0))
            sigma_r = radius
            mad_factor = float(self.params.get("nb_mad_factor", 10.0))
            angle_factor = float(self.params.get("nb_angle_factor", 8.0))
            min_neighbors = int(self.params.get("nb_min_neighbors", 5))
    
            (point_keep,
             comp_dev,
             dir_dev,
             comp_flag,
             dir_flag) = self._fast_filter_track_points_combined_kdtree(
                x_plot, y_plot,
                x_plotv, y_plotv,
                t_plot,
                dt=dt,
                radius=radius,
                sigma_r=sigma_r,
                mad_factor=mad_factor,
                angle_factor=angle_factor,
                min_neighbors=min_neighbors
            )
    
            self.point_keep = point_keep
            self.comp_dev = comp_dev
            self.dir_dev = dir_dev
            self.comp_flag = comp_flag
            self.dir_flag = dir_flag

            for i in range(n_tracks):
                if len(point_keep[i]) == len(flag_points[i]):
                    flag_points[i] = flag_points[i] & point_keep[i]
    
            kept_pts = int(sum(int(np.sum(m)) for m in point_keep))
            self._status_log_append(
                f"Neighborhood filter applied: radius={radius:g}, dt={dt:g}us, "
                f"min_neighbors={min_neighbors}, kept_points_total={kept_pts}."
            )
        else:
            self._status_log_append("Neighborhood filter disabled.")
        self.flag_points = flag_points

        self.flag_track = np.array([bool(np.any(fp)) for fp in flag_points], dtype=bool)
    
        kept_tracks = int(np.sum(self.flag_track))
        removed_tracks = int(n_tracks - kept_tracks)
    
        filters = []
        if apply_quality: filters.append("quality")
        if apply_nb: filters.append("neighborhood")
        if not filters: filters.append("none")
    
        self._set_run_status(
            f"Validation done ({'+'.join(filters)}): kept {kept_tracks}/{n_tracks}, removed {removed_tracks}."
        )
        self._status_log_append(
            f"Run validation ({'+'.join(filters)}): kept {kept_tracks}/{n_tracks}, removed {removed_tracks}."
        )
    
        self._update_validation_overlay()
        
        total = sum(len(fp) for fp in self.flag_points)
        bad = sum(int(np.sum(~np.asarray(fp, dtype=bool))) for fp in self.flag_points)
        self._status_log_append(f"Neighborhood: invalid points = {bad}/{total}")
        
    def _update_validation_overlay(self) -> None:
        """Draw validation results as green/red line segments on pseudo-frame (tracking-like)."""
        if not dpg.does_item_exist(self._pf_drawlist_tag):
            return

        try:
            if dpg.does_item_exist(self._pf_tracks_node_tag):
                dpg.delete_item(self._pf_tracks_node_tag)
        except Exception:
            pass
        with dpg.draw_node(tag=self._pf_tracks_node_tag, parent=self._pf_drawlist_tag):
            pass
    
        if self.preprocessing_tab is None or self.tracking_tab is None:
            return
        if getattr(self.tracking_tab, "last_result", None) is None:
            return

        flag_points = getattr(self, "flag_points", None)
        if flag_points is None:
            flag_points = getattr(self, "flag", None)
        if flag_points is None:
            return
    
        out = self.tracking_tab.last_result
    
        def _get(name: str):
            v = getattr(out, name, None)
            if v is None and isinstance(out, dict):
                v = out.get(name)
            return v
    
        x_plot = _get("x_plot")
        y_plot = _get("y_plot")
        t_plot = _get("t_plot")
    
        if x_plot is None or y_plot is None or t_plot is None:
            return

        if not isinstance(x_plot, (list, tuple)): x_plot = [x_plot]
        if not isinstance(y_plot, (list, tuple)): y_plot = [y_plot]
        if not isinstance(t_plot, (list, tuple)): t_plot = [t_plot]
    
        n_tracks = min(len(x_plot), len(y_plot), len(t_plot), len(flag_points))
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
    
        T_min = float(t0) - (factor - 1.0) * acc_us
        T_max = float(t0) + acc_us
    
        green = (0, 220, 0, 255)
        red = (220, 0, 0, 255)

        for i in range(n_tracks):
            x = np.asarray(x_plot[i], dtype=float).ravel() - 1.0
            y = np.asarray(y_plot[i], dtype=float).ravel() - 1.0
            t = np.asarray(t_plot[i], dtype=float).ravel()
    
            fp = np.asarray(flag_points[i], dtype=bool).ravel()

            n = min(len(x), len(y), len(t), len(fp))
            if n < 2:
                continue
            x = x[:n]; y = y[:n]; t = t[:n]; fp = fp[:n]
    
            m = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
            m &= (t >= T_min) & (t <= T_max)
            if np.count_nonzero(m) < 2:
                continue

            xw = x[m]; yw = y[m]; fpw = fp[m]
            for k in range(len(xw) - 1):
                col = green if (bool(fpw[k]) and bool(fpw[k + 1])) else red
                dpg.draw_line(
                    p1=[float(xw[k]), float(yw[k])],
                    p2=[float(xw[k + 1]), float(yw[k + 1])],
                    color=col,
                    thickness=2.0,
                    parent=self._pf_tracks_node_tag,
                )

    def _set_run_status(self, msg: str) -> None:
        if dpg.does_item_exist(self._run_status_tag):
            dpg.set_value(self._run_status_tag, msg)
            
    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def _weighted_median_fast(values, weights):
        idx = np.argsort(values)
        cw = np.cumsum(weights[idx])
        return values[idx[np.searchsorted(cw, 0.5 * cw[-1])]]
    
    @classmethod
    def _weighted_mad_fast(cls, values, weights, median):
        return cls._weighted_median_fast(np.abs(values - median), weights)
    
    @classmethod
    def _weighted_circular_median_fast(cls, angles, weights):
        diff = cls._angle_diff(angles[:, None], angles[None, :])
        costs = np.sum(weights[None, :] * np.abs(diff), axis=1)
        return angles[np.argmin(costs)]
    
    @classmethod
    def _circular_mad_fast(cls, angles, weights, median_angle):
        dev = np.abs(cls._angle_diff(angles, median_angle))
        idx = np.argsort(dev)
        cw = np.cumsum(weights[idx])
        return dev[idx[np.searchsorted(cw, 0.5 * cw[-1])]]
    
    def _fast_filter_track_points_combined_kdtree(
        self,
        x_plot, y_plot,
        x_plotv, y_plotv,
        t_plot,
        *,
        dt=0.1,
        radius=5.0,
        sigma_r=None,
        mad_factor=3.5,
        angle_factor=3.0,
        min_neighbors=5
    ):
        n_tracks = len(x_plot)
        point_keep = [np.ones(len(t_plot[i]), dtype=bool) for i in range(n_tracks)]
        comp_deviation = [np.zeros(len(t_plot[i])) for i in range(n_tracks)]
        dir_deviation  = [np.zeros(len(t_plot[i])) for i in range(n_tracks)]
        comp_flag = [np.zeros(len(t_plot[i]), dtype=bool) for i in range(n_tracks)]
        dir_flag  = [np.zeros(len(t_plot[i]), dtype=bool) for i in range(n_tracks)]
    
        valid_tracks = [i for i in range(n_tracks) if len(t_plot[i]) > 0]
        if not valid_tracks:
            return point_keep, comp_deviation, dir_deviation, comp_flag, dir_flag
    
        if sigma_r is None:
            sigma_r = radius
        sigma_r2 = float(sigma_r) * float(sigma_r)

        tmins = []
        tmaxs = []
        for i in valid_tracks:
            ti = np.asarray(t_plot[i], dtype=float)
            ti = ti[np.isfinite(ti)]
            if ti.size == 0:
                continue
            tmins.append(np.min(ti))
            tmaxs.append(np.max(ti))
        
        if not tmins:
            return point_keep, comp_deviation, dir_deviation, comp_flag, dir_flag
        
        t_min = float(np.min(tmins))
        t_max = float(np.max(tmaxs))
        
        time_bins = np.arange(t_min, t_max + dt, dt)
        for t0 in time_bins:
            t1 = t0 + dt
    
            pos, vel, index_map, track_ids = [], [], [], []
    
            for i in valid_tracks:
                ti = np.asarray(t_plot[i], dtype=float)
                mask = (ti >= t0) & (ti < t1)
                idxs = np.where(mask)[0]
                for k in idxs:
                    vx = float(x_plotv[i][k])
                    vy = float(y_plotv[i][k])
                    pos.append((float(x_plot[i][k]), float(y_plot[i][k])))
                    vel.append((vx, vy))
                    index_map.append((i, int(k)))
                    track_ids.append(i)
    
            if len(pos) < min_neighbors:
                continue
    
            pos = np.asarray(pos, dtype=float)
            vel = np.asarray(vel, dtype=float)
            angles = np.arctan2(vel[:, 1], vel[:, 0])
            track_ids = np.asarray(track_ids, dtype=int)
    
            tree = cKDTree(pos)
    
            for idx, (ti, ki) in enumerate(index_map):
                neigh = tree.query_ball_point(pos[idx], radius)
                neigh = [j for j in neigh if j != idx and track_ids[j] != ti]
                if len(neigh) < min_neighbors:
                    continue
    
                neigh = np.asarray(neigh, dtype=int)
                d = np.linalg.norm(pos[neigh] - pos[idx], axis=1)
                weights = np.exp(-0.5 * (d * d) / sigma_r2)

                vx = vel[neigh, 0]
                vy = vel[neigh, 1]
    
                med_vx = self._weighted_median_fast(vx, weights)
                med_vy = self._weighted_median_fast(vy, weights)
                mad_vx = self._weighted_mad_fast(vx, weights, med_vx)
                mad_vy = self._weighted_mad_fast(vy, weights, med_vy)
    
                sigma_vx = 1.4826 * mad_vx if mad_vx else np.inf
                sigma_vy = 1.4826 * mad_vy if mad_vy else np.inf
    
                dev_vx = abs(vel[idx, 0] - med_vx) / sigma_vx
                dev_vy = abs(vel[idx, 1] - med_vy) / sigma_vy
                comp_outlier = (dev_vx > mad_factor) or (dev_vy > mad_factor)

                neigh_angles = angles[neigh]
                med_angle = self._weighted_circular_median_fast(neigh_angles, weights)
                mad_angle = self._circular_mad_fast(neigh_angles, weights, med_angle)
                sigma_angle = 1.4826 * mad_angle if mad_angle else np.inf
    
                dev_angle = abs(self._angle_diff(angles[idx], med_angle)) / sigma_angle
                dir_outlier = dev_angle > angle_factor
    
                comp_deviation[ti][ki] = max(dev_vx, dev_vy)
                dir_deviation[ti][ki] = dev_angle
                comp_flag[ti][ki] = comp_outlier
                dir_flag[ti][ki] = dir_outlier
    
                if comp_outlier or dir_outlier:
                    point_keep[ti][ki] = False
    
        return point_keep, comp_deviation, dir_deviation, comp_flag, dir_flag
    
    @staticmethod
    def _apply_point_mask(data, masks):
        return [d[m] for d, m in zip(data, masks)]

    def browse_vali_save_results_file(self, sender=None, app_data=None, user_data=None):
        if dpg.does_item_exist("vali_save_results_file_dialog"):
            dpg.show_item("vali_save_results_file_dialog")
    
    def on_vali_save_results_file_selected(self, sender, app_data, user_data=None):
        path = str(app_data.get("file_path_name", "")).strip()
        if path:
            dpg.set_value("vali_save_results_file_input", path)
    
    def _build_validation_payload_bytes(self) -> tuple[bytes, bytes]:
        if self.tracking_tab is None or getattr(self.tracking_tab, "last_result", None) is None:
            raise ValueError("No tracking results available. Run Tracking first.")

        flag_points = getattr(self, "flag_points", None)
        if flag_points is None:
            raise ValueError("No validation flags available. Run Validation first.")
    
        src = self.tracking_tab.last_result
        def _get(obj, name):
            v = getattr(obj, name, None)
            if v is None and isinstance(obj, dict):
                v = obj.get(name)
            return v
    
        x_plot = _get(src, "x_plot")
        y_plot = _get(src, "y_plot")
        x_plotv = _get(src, "x_plotv")
        y_plotv = _get(src, "y_plotv")
        t_plot = _get(src, "t_plot")
    
        if any(v is None for v in (x_plot, y_plot, x_plotv, y_plotv, t_plot)):
            raise ValueError("Tracking output incomplete (missing x/y/v/t).")

        if not isinstance(x_plot, (list, tuple)): x_plot = [x_plot]
        if not isinstance(y_plot, (list, tuple)): y_plot = [y_plot]
        if not isinstance(x_plotv, (list, tuple)): x_plotv = [x_plotv]
        if not isinstance(y_plotv, (list, tuple)): y_plotv = [y_plotv]
        if not isinstance(t_plot, (list, tuple)): t_plot = [t_plot]
    
        n_tracks = min(len(x_plot), len(y_plot), len(x_plotv), len(y_plotv), len(t_plot), len(flag_points))
        if n_tracks == 0:
            raise ValueError("No tracks to save (n_tracks=0).")

        x_plot_s, y_plot_s, t_plot_s = [], [], []
        x_plotv_s, y_plotv_s = [], []
    
        for i in range(n_tracks):
            xi = np.asarray(x_plot[i], dtype=float).copy()
            yi = np.asarray(y_plot[i], dtype=float).copy()
            ti = np.asarray(t_plot[i], dtype=float).copy()
            vxi = np.asarray(x_plotv[i], dtype=float).copy()
            vyi = np.asarray(y_plotv[i], dtype=float).copy()
    
            fp = np.asarray(flag_points[i], dtype=bool).ravel()
            n = min(len(xi), len(yi), len(ti), len(vxi), len(vyi), len(fp))
            if n == 0:
                x_plot_s.append(xi)
                y_plot_s.append(yi)
                t_plot_s.append(ti)
                x_plotv_s.append(vxi)
                y_plotv_s.append(vyi)
                continue
    
            xi = xi[:n]; yi = yi[:n]; ti = ti[:n]; vxi = vxi[:n]; vyi = vyi[:n]; fp = fp[:n]
    
            bad = ~fp
            xi[bad] = np.nan
            yi[bad] = np.nan
            ti[bad] = np.nan
            vxi[bad] = np.nan
            vyi[bad] = np.nan
    
            x_plot_s.append(xi)
            y_plot_s.append(yi)
            t_plot_s.append(ti)
            x_plotv_s.append(vxi)
            y_plotv_s.append(vyi)
        result_out = deepcopy(src)
        def _set(obj, name, value):
            if isinstance(obj, dict):
                obj[name] = value
            else:
                setattr(obj, name, value)
    
        _set(result_out, "x_plot", x_plot_s)
        _set(result_out, "y_plot", y_plot_s)
        _set(result_out, "t_plot", t_plot_s)
        _set(result_out, "x_plotv", x_plotv_s)
        _set(result_out, "y_plotv", y_plotv_s)
        _set(result_out, "validation_flag_points", flag_points)
        if getattr(self, "flag_track", None) is not None:
            _set(result_out, "validation_flag_track", self.flag_track)

        ctx = getattr(self.tracking_tab, "_last_run_ctx", None)
        if ctx is None:
            ctx = {}

        if hasattr(self.tracking_tab, "_sanitize_tracking_ctx"):
            ctx2 = self.tracking_tab._sanitize_tracking_ctx(ctx)
        else:
            ctx2 = ctx
    
        result_bytes = pickle.dumps(result_out, protocol=pickle.HIGHEST_PROTOCOL)
        ctx_bytes = pickle.dumps(ctx2, protocol=pickle.HIGHEST_PROTOCOL)
        return result_bytes, ctx_bytes
    
    def save_validation_results(self, sender=None, app_data=None, user_data=None):
        try:
            path = str(dpg.get_value("vali_save_results_file_input") or "").strip()
            ext = str(dpg.get_value("vali_save_results_format") or ".npz").strip().lower()
    
            if not path:
                self.browse_vali_save_results_file()
                return
    
            p = Path(path)
            if p.suffix.lower() not in [".npz", ".mat"]:
                p = p.with_suffix(ext)
    
            result_bytes, ctx_bytes = self._build_validation_payload_bytes()
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
    
            dpg.set_value("vali_save_results_status", f"Saved: {p}")
            self._status_log_append(f"Saved validated tracks: {p}")
        except Exception as e:
            dpg.set_value("vali_save_results_status", f"Save failed: {e}")
            self._status_log_append(f"Save failed: {e}")