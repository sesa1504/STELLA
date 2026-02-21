import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

import dearpygui.dearpygui as dpg
import numpy as np
from config.constants import (
    DEFAULT_PARAMS,
    PSEUDOFAME_DIMENSIONS,
    SUPPORTED_FORMATS,
)
from utils.dataloader import build_preview_and_stats, load_data_file
from utils.helpers import EventDataHelper
from utils.plots import (
    figure_to_rgba_flat,
)

class PreprocessingTab:
    label: str = "Preprocessing"

    def __init__(self) -> None:
        self.params: dict[str, Any] = deepcopy(DEFAULT_PARAMS)
        self.data_file: str = ""
        self.output_dir: str = ""
        self.supported_formats: Sequence[str] = SUPPORTED_FORMATS
        self.pseudoframe_texture_tag: str = "preprocessing_pseudoframe_texture"
        self.pseudoframe_dimensions = PSEUDOFAME_DIMENSIONS
        self.last_loaded_data: Any = None
        self.display_t0: float | None = None
        self.raw_events: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = None
        self.filtered_events: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = None
        self.last_loaded_file_ext: Optional[str] = None
        self.params = {
            "accumulation_time_ms": 2.0,
        }
        self.last_event_points: Optional[
                tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
            ] = None
        self.main_texture_tag: str = "preprocessing_main_plot_texture"
        self._pseudoframe_status_tags: set[str] = {"preprocessing_pseudoframe_status"}
        self._pseudoframe_time_info_tags: set[str] = {"pseudoframe_time_info"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, parent: int) -> None:
        """Build the tab UI."""
        self._ensure_textures()
        with dpg.group(parent=parent):
            with dpg.group(horizontal=True):
                with dpg.child_window(label="Data Management", width=500, height=950):
                    self._build_data_panel()
                with dpg.child_window(label="Visualization", width=1920-550, height=950):
                    dpg.add_text("VISUALIZATION", color=(100, 200, 255))
                    dpg.add_separator()
                    self._build_preprocessing_parameters_panel()
        self.reset_pseudoframe_visualization()

    def register_pseudoframe_mirror(self, *, status_tag: str, time_info_tag: str) -> None:
        if status_tag:
            self._pseudoframe_status_tags.add(status_tag)
        if time_info_tag:
            self._pseudoframe_time_info_tags.add(time_info_tag)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _ensure_textures(self) -> None:
        if not dpg.does_item_exist(self.main_texture_tag):
            dummy_data = np.zeros((580, 700, 4), dtype=np.float32)
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=500,
                    height=580,
                    default_value=dummy_data.flatten(),
                    format=dpg.mvFormat_Float_rgba,
                    tag=self.main_texture_tag,
                )
                pseudo_width, pseudo_height = self.pseudoframe_dimensions
                pseudo_data = np.zeros((pseudo_height, pseudo_width, 4), dtype=np.float32)
                dpg.add_raw_texture(
                    width=pseudo_width,
                    height=pseudo_height,
                    default_value=pseudo_data.flatten(),
                    format=dpg.mvFormat_Float_rgba,
                    tag=self.pseudoframe_texture_tag,
                )

    def _build_data_panel(self) -> None:
        dpg.add_text("DATA MANAGEMENT", color=(100, 200, 255))
        dpg.add_separator()
        dpg.add_text("Select event stream file:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                default_value="",
                tag="data_file_input",
                width=300,
                hint="Paste path and press Enter or click Browse...",
                callback=self.on_data_file_path_typed,
                on_enter=True,
            )
            dpg.add_button(label="Browse", callback=self.browse_data_file, width=100)
        formats_text = " | ".join(self.supported_formats)
        dpg.add_text("Supported File Types:", color=(150, 200, 150))
        dpg.add_text(formats_text, color=(180, 180, 180), wrap=700)
        dpg.add_button(label="load data", callback=self.validate_file, width=100)
        dpg.add_text("", tag="validation_status", color=(100, 255, 100))
        dpg.add_separator()
        dpg.add_text("DATA PREVIEW", color=(100, 200, 255))
        dpg.add_separator()
        with dpg.child_window(label="Data Preview", height=150, width=480):
            dpg.add_text("", tag="data_preview_text", wrap=700)
        dpg.add_separator()
        dpg.add_text("PREPROCESSING PARAMETERS", color=(100, 200, 255))
        dpg.add_separator()

        dpg.add_text("Accumulation time:", color=(200, 200, 100))
        dpg.add_input_float(
            label="accumulation time (ms)",
            tag="bracket_preprocessing_accumulation_time",
            default_value=float(self.params["accumulation_time_ms"]),
            width=150,
            on_enter=True,
            min_value=0.0,
            min_clamped=True,
            callback=self.update_preprocessing_accumulation_time,
        )
        dpg.add_text("ROI:", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="xmin",
                tag="roi_xmin",
                default_value=int(self.params.get("roi_xmin", 0)),
                width=170,
                on_enter=True,
                callback=self.update_roi_param,
                user_data="roi_xmin",
            )
            dpg.add_input_int(
                label="xmax",
                tag="roi_xmax",
                default_value=int(self.params.get("roi_xmax", 1279)),
                width=170,
                on_enter=True,
                callback=self.update_roi_param,
                user_data="roi_xmax",
            )
        
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="ymin",
                tag="roi_ymin",
                default_value=int(self.params.get("roi_ymin", 0)),
                width=170,
                on_enter=True,
                callback=self.update_roi_param,
                user_data="roi_ymin",
            )
            dpg.add_input_int(
                label="ymax",
                tag="roi_ymax",
                default_value=int(self.params.get("roi_ymax", 719)),
                width=170,
                on_enter=True,
                callback=self.update_roi_param,
                user_data="roi_ymax",
            )
            
        dpg.add_separator()
        dpg.add_text("Time range:", color=(200, 200, 100))
        dpg.add_checkbox(
                label="Enable time crop",
                tag="time_crop_enabled",
                default_value=bool(self.params.get("time_crop_enabled", False)),
                callback=self.on_time_crop_toggle,
            )
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="tmin",
                tag="time_tmin",
                default_value=float(self.params.get("tmin", 0.0)),
                width=170,
                on_enter=True,
                callback=self.update_time_param,
                user_data="tmin",
            )
            dpg.add_input_float(
                label="tmax",
                tag="time_tmax",
                default_value=float(self.params.get("tmax", 0.0)),
                width=170,
                on_enter=True,
                callback=self.update_time_param,
                user_data="tmax",
            )
        
        dpg.add_separator()
        dpg.add_text("EXPORT FILTERED EVENTS", color=(100, 200, 255))

        dpg.add_text("Save filtered events (X,Y,T,P):", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                default_value="",
                tag="save_filtered_file_input",
                width=300,
                hint="Choose target file (.npz or .mat)",
                callback=self.on_save_filtered_path_typed,
                on_enter=True,
            )
            dpg.add_button(label="Browse", callback=self.browse_save_filtered_file, width=80)
            dpg.add_combo(
                items=[".npz", ".mat"],
                default_value=".npz",
                tag="save_filtered_format",
                width=70,
            )
        dpg.add_button(label="Save data", callback=self.save_filtered_events, width=120)
        dpg.add_text("", tag="save_filtered_status", color=(100, 255, 100))

        if not dpg.does_item_exist("save_filtered_file_dialog"):
            with dpg.file_dialog(
                tag="save_filtered_file_dialog",
                label="Save filtered events",
                directory_selector=False,
                show=False,
                callback=self.on_save_filtered_file_selected,
                cancel_callback=lambda s, a: dpg.set_value("save_filtered_status", "Save cancelled"),
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))
            
            
        if not dpg.does_item_exist("data_file_dialog"):
            with dpg.file_dialog(
                tag="data_file_dialog",
                label="Select event-stream file",
                directory_selector=False,
                show=False,
                callback=self.on_data_file_selected,
                cancel_callback=self.on_data_file_cancelled,
                width=800,
                height=500,
            ):
                dpg.add_file_extension(".npz", color=(150, 255, 150, 255))
                dpg.add_file_extension(".mat", color=(150, 255, 150, 255))
                dpg.add_file_extension(".raw", color=(150, 255, 150, 255))

    def _build_preprocessing_parameters_panel(self) -> None:
        try:
            if dpg.does_item_exist("preprocessing_parameters_display"):
                dpg.delete_item("preprocessing_parameters_display")
        except Exception:
            pass
        dpg.add_image(self.pseudoframe_texture_tag, width=self.pseudoframe_dimensions[0], height=self.pseudoframe_dimensions[1], tag="preprocessing_pseudoframe_image")
        dpg.add_text("Load data to generate pseudo-frame.", tag="preprocessing_pseudoframe_status", wrap=730)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="<< Prev", width=120, callback=self.on_prev_pseudoframe)
            dpg.add_button(label="Next >>", width=120, callback=self.on_next_pseudoframe)
            dpg.add_text("", tag="pseudoframe_time_info", wrap=600)

        dpg.add_separator()
        dpg.add_text("Status Log", color=(200, 200, 100))
        with dpg.child_window(label="Status Log", height=90, width=720):
            dpg.add_input_text(tag="status_log_text", multiline=True, readonly=True, width=695, height=75)

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def browse_data_file(self, sender=None, app_data=None, user_data=None):
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        if data_dir.exists():
            if dpg.does_item_exist("data_file_dialog"):
                dpg.configure_item("data_file_dialog", default_path=str(data_dir.resolve()))
    
        dpg.show_item("data_file_dialog")
    
    def on_data_file_cancelled(self, sender, app_data):
        self.log_status("File selection cancelled")

    def on_data_file_selected(self, sender, app_data) -> None: 
        if app_data.get("file_path_name"):
            self.data_file = app_data["file_path_name"]
            dpg.set_value("data_file_input", self.data_file)
            if dpg.does_item_exist("preprocessing_file_display"):
                dpg.set_value("preprocessing_file_display", f"{Path(self.data_file).name}")
            dpg.set_value("validation_status", "File selected. Click 'load data' to verify.")
            self.log_status(f"File selected: {self.data_file}")
            self.update_preprocessing_parameters_display()
            
    def on_data_file_path_typed(self, sender, app_data) -> None:
        path = (app_data or "").strip().strip('"').strip("'")
        if not path:
            self.data_file = ""
            if dpg.does_item_exist("validation_status"):
                dpg.set_value("validation_status", "No file selected")
            self.log_status("File path cleared")
            return
    
        self.data_file = path
        dpg.set_value("data_file_input", self.data_file)
    
        if dpg.does_item_exist("validation_status"):
            dpg.set_value("validation_status", "File path entered. Click 'load data' to verify.")
    
        self.log_status(f"File path entered: {self.data_file}")
        self.update_preprocessing_parameters_display()

    def validate_file(self, sender=None, app_data=None, user_data=None) -> None:
        if not self.data_file:
            dpg.set_value("validation_status", "No file selected")
            return
        try:
            dpg.set_value("validation_status", "loading...")
            loaded = load_data_file(self.data_file)

            if loaded.width is not None:
                self.params["width"] = loaded.width
            if loaded.height is not None:
                self.params["height"] = loaded.height

            self.last_loaded_data = loaded.data
            self.last_loaded_file_ext = loaded.file_ext
            self.load_data_preview()
            dpg.set_value("validation_status", loaded.status_message)
            self.log_status(f"File validated successfully: {self.data_file}")
        except Exception as exc:
            dpg.set_value("validation_status", f"Error: {exc}")
            self.log_status(f"File validation failed: {exc}")

    def load_data_preview(self) -> None:
        if not self.data_file:
            return
        try:
            loaded = load_data_file(self.data_file)
            data = loaded.data
            file_ext = loaded.file_ext

            self.update_pseudoframe_visualization(None, "Generating pseudo-frame preview...")

            preview_text = build_preview_and_stats(loaded)
            dpg.set_value("data_preview_text", preview_text)

            self.update_preprocessing_parameters_display()

            events = EventDataHelper.extract_event_points(file_ext, data, self.data_file)
            self.raw_events = events
            xs, ys, ts, ps = self.raw_events

            tmin = float(np.min(ts))
            tmax = float(np.max(ts))
            
            self.params["tmin"] = tmin
            self.params["tmax"] = tmax
            
            if dpg.does_item_exist("time_tmin"):
                dpg.set_value("time_tmin", tmin)
            if dpg.does_item_exist("time_tmax"):
                dpg.set_value("time_tmax", tmax)
    
            if "roi_xmax" not in self.params:
                xs, ys, ts, ps = self.raw_events
                self.params["roi_xmin"] = int(np.min(xs))
                self.params["roi_xmax"] = int(np.max(xs))
                self.params["roi_ymin"] = int(np.min(ys))
                self.params["roi_ymax"] = int(np.max(ys))
                for k in ("roi_xmin","roi_xmax","roi_ymin","roi_ymax"):
                    if dpg.does_item_exist(k):
                        dpg.set_value(k, self.params[k])
            
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Data loaded, ROI applied.")
            
            xs, ys, ts, ps = self.filtered_events if self.filtered_events else self.raw_events
            self.display_t0 = float(np.min(ts))
        except Exception as exc:
            dpg.set_value("data_preview_text", f"Error loading preview: {exc}")
            self.log_status(f"Error loading data preview: {exc}")

    def apply_event_filters(self) -> None:
        if not getattr(self, "raw_events", None):
            self.filtered_events = None
            return
    
        xs, ys, ts, ps = self.raw_events
    
        xmin = int(self.params.get("roi_xmin", 0))
        xmax = int(self.params.get("roi_xmax", 1279))
        ymin = int(self.params.get("roi_ymin", 0))
        ymax = int(self.params.get("roi_ymax", 719))
    
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
    
        X = np.asarray(xs).astype(np.int64)
        Y = np.asarray(ys).astype(np.int64)
        T = np.asarray(ts).astype(np.float64)
        P = np.asarray(ps) if ps is not None else None
    
        mask = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
    
        if bool(self.params.get("time_crop_enabled", False)):
            tmin = self.params.get("tmin", None)
            tmax = self.params.get("tmax", None)

            if tmin is not None and str(tmin) != "":
                tmin_f = float(tmin)
                mask = mask & (T >= tmin_f)
    
            if tmax is not None and str(tmax) != "":
                tmax_f = float(tmax)
                mask = mask & (T <= tmax_f)
    
            if tmin is not None and tmax is not None and str(tmin) != "" and str(tmax) != "":
                tmin_f = float(tmin)
                tmax_f = float(tmax)
                if tmax_f < tmin_f:
                    tmin_f, tmax_f = tmax_f, tmin_f
                    mask = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
                    mask = mask & (T >= tmin_f) & (T <= tmax_f)
                    
        if self.params.get("use_only_pos", True) and P is not None:
            mask = mask & (P > 0)
    
        Xf = X[mask]
        Yf = Y[mask]
        Tf = T[mask]
        Pf = P[mask] if P is not None else None
    
        self.filtered_events = (Xf, Yf, Tf, Pf)
        self.log_status(f"Filters applied: kept {Xf.size} / {X.size} events")
        
    def update_roi_param(self, sender, app_data, user_data):
        self.params[user_data] = int(app_data)

        if self.raw_events:
            self.apply_event_filters()
            self.update_pseudoframe_from_data("ROI updated.")
            
    def update_time_param(self, sender, app_data, user_data):
        self.params[user_data] = float(app_data)
        if self.params.get("time_crop_enabled", False) and getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Time crop updated.")
            
    def on_time_crop_toggle(self, sender, app_data):
        self.params["time_crop_enabled"] = bool(app_data)
        if getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Time crop toggled.")

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def update_preprocessing_param(self, sender=None, app_data=None, user_data=None) -> None:  
        param_name = user_data
        if param_name in ("width", "height"):
            value = dpg.get_value(f"bracket_preprocessing_{param_name}")
            self.params[param_name] = value
            self.log_status(f"Preprocessing {param_name} updated to {value}")
            self.update_preprocessing_parameters_display()
            if self.last_event_points:
                self.update_pseudoframe_from_data(f"Pseudo-frame updated ({param_name} changed).")
                self.show_raw_events_in_visualization()
     
    def update_preprocessing_accumulation_time(self, sender, app_data, user_data=None):
        try:
            acc_ms = float(app_data)
        except Exception:
            return

        self.params["accumulation_time_ms"] = acc_ms
        step = acc_ms * 1000.0  
        self.log_status(f"Accumulation time set to {acc_ms:.3f} ms -> window step {step:.0f} ts-units")

        if getattr(self, "raw_events", None):
            self.apply_event_filters()
            self.update_pseudoframe_from_data("Accumulation time changed.")

    def update_preprocessing_parameters_display(self) -> None:
        if self.data_file:
            file_display = f"{Path(self.data_file).name}"
            if dpg.does_item_exist("preprocessing_file_display"):
                dpg.set_value("preprocessing_file_display", file_display)
        else:
            if dpg.does_item_exist("preprocessing_file_display"):
                dpg.set_value("preprocessing_file_display", "No file selected")
        
        if not dpg.does_item_exist("preprocessing_parameters_display"):
            return
        params_text = "Current Settings:\n"
        params_text += f"  Width: {self.params.get('width', 720)}\n"
        params_text += f"  Height: {self.params.get('height', 1280)}\n"
        acc = self.params.get("accumulation_time_ms", None)
        if acc is not None and acc != "":
            params_text += f"  Accumulation Time: {acc} ms\n"
        params_text += f"  Use Only Positive: {self.params.get('use_only_pos', True)}\n"
        dpg.set_value("preprocessing_parameters_display", params_text)

    # ------------------------------------------------------------------
    # Pseudo frame helpers
    # ------------------------------------------------------------------
    def reset_pseudoframe_visualization(self, message: str = "Load data to generate pseudo-frame.") -> None:
        pseudo_width, pseudo_height = self.pseudoframe_dimensions
        blank = np.zeros((pseudo_height, pseudo_width, 4), dtype=np.float32)
        if dpg.does_item_exist(self.pseudoframe_texture_tag):
            dpg.set_value(self.pseudoframe_texture_tag, blank.flatten())
        for tag in list(self._pseudoframe_status_tags):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, message)

    def update_pseudoframe_visualization(self, frame: Optional[np.ndarray], status: Optional[str] = None) -> None:
        if frame is None:
            self.reset_pseudoframe_visualization(status or "Pseudo-frame preview not available for this format.")
            return
        self.render_pseudoframe_to_texture(frame)
        if status:
            for tag in list(self._pseudoframe_status_tags):
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, status)

    def render_pseudoframe_to_texture(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        frame_array = np.array(frame, dtype=np.float32)
        frame_array = np.nan_to_num(frame_array, nan=0.0, posinf=0.0, neginf=0.0)
            
        if frame_array.ndim == 2:
            pseudo_width, pseudo_height = self.pseudoframe_dimensions
            src_height, src_width = frame_array.shape
            if src_height <= 0 or src_width <= 0:
                return
        
            y_indices = np.linspace(0, src_height - 1, pseudo_height).astype(np.int32)
            x_indices = np.linspace(0, src_width - 1, pseudo_width).astype(np.int32)
            resized = frame_array[y_indices][:, x_indices]

            cmap = np.array([
                [30,  37,  52],
                [64, 124, 198],
                [220, 226, 238],
            ], dtype=np.float32) / 255.0

            mask = resized > 0
            rgba = np.zeros((pseudo_height, pseudo_width, 4), dtype=np.float32)
            rgba[:, :, :3] = cmap[0]          
            rgba[mask, :3] = cmap[2]          
            rgba[:, :, 3] = 1.0               
        elif frame_array.ndim == 3 and frame_array.shape[2] in (3, 4):
            pseudo_width, pseudo_height = self.pseudoframe_dimensions
            src_height, src_width = frame_array.shape[:2]
            if src_height <= 0 or src_width <= 0:
                return
            y_indices = np.linspace(0, src_height - 1, pseudo_height).astype(np.int32)
            x_indices = np.linspace(0, src_width - 1, pseudo_width).astype(np.int32)
            resized = frame_array[y_indices][:, x_indices]
            rgba = np.zeros((pseudo_height, pseudo_width, 4), dtype=np.float32)
            rgba[:, :, :resized.shape[2]] = resized[:, :, :resized.shape[2]]
            if resized.shape[2] == 3:
                rgba[:, :, 3] = 1.0
            else:
                rgba[:, :, 3] = resized[:, :, 3]
        else:
            return
        if dpg.does_item_exist(self.pseudoframe_texture_tag):
            dpg.set_value(self.pseudoframe_texture_tag, rgba.flatten())

    def update_pseudoframe_from_data(self, message: Optional[str] = None) -> None:
        if not self.filtered_events:
            self.update_pseudoframe_visualization(None, "Pseudo-frame preview unavailable (no events / ROI empty).")
            self.log_status("Cannot generate pseudo-frame: no filtered events available")
            return
        try:
            xs, ys, ts, polarities = self.filtered_events
            if xs is None or ys is None or len(xs) == 0 or len(ys) == 0:
                self.update_pseudoframe_visualization(None, "Pseudo-frame preview unavailable (empty coordinates).")
                self.log_status("Cannot generate pseudo-frame: empty coordinate arrays")
                return
            width = 1280
            height = 720
            
            ts_arr = np.asarray(ts).flatten()
            xs_arr = np.asarray(xs).flatten()
            ys_arr = np.asarray(ys).flatten()
            
            n = min(xs_arr.size, ys_arr.size, ts_arr.size)
            xs_arr, ys_arr, ts_arr = xs_arr[:n], ys_arr[:n], ts_arr[:n]
            
            m = np.isfinite(xs_arr) & np.isfinite(ys_arr) & np.isfinite(ts_arr)
            xs_arr, ys_arr, ts_arr = xs_arr[m], ys_arr[m], ts_arr[m]
            
            if xs_arr.size == 0:
                self.update_pseudoframe_visualization(None, "No valid events for accumulation.")
                return
            
            acc_ms = float(self.params.get("accumulation_time_ms", 2.0))
            step = acc_ms * 1000.0  
            
            if self.display_t0 is None:
                self.display_t0 = float(np.min(ts_arr))
            
            T_min = float(self.display_t0)
            T_max = T_min + step
            
            mask = (ts_arr > T_min) & (ts_arr <= T_max)
            X = xs_arr[mask].astype(np.int64)
            Y = ys_arr[mask].astype(np.int64)

            valid = (X >= 0) & (X < width) & (Y >= 0) & (Y < height)
            X, Y = X[valid], Y[valid]
            
            I = np.zeros((height, width), dtype=np.float32)
            I[Y.astype(np.intp), X.astype(np.intp)] = 1.0

            self.update_pseudoframe_visualization(I, f"Accumulation preview ({int(mask.sum())} events).")
            for tag in list(self._pseudoframe_time_info_tags):
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, f"t: [{T_min:.0f} .. {T_max:.0f}] (display window)")
        except Exception as exc:
            error_message = f"Pseudo-frame error: {exc}"
            self.log_status(error_message)
            self.update_pseudoframe_visualization(None, error_message)

    def _get_acc_step_ts_units(self) -> float:
        acc_ms = float(self.params.get("accumulation_time_ms", 2.0))
        return acc_ms * 1000.0
    
    def on_next_pseudoframe(self, sender=None, app_data=None, user_data=None):
        self._shift_time_window(+1)
    
    def on_prev_pseudoframe(self, sender=None, app_data=None, user_data=None):
        self._shift_time_window(-1)
    
    def _shift_time_window(self, direction: int) -> None:
        if not self.filtered_events:
            self.log_status("No filtered events available.")
            return
    
        xs, ys, ts, ps = self.filtered_events
        acc_ms = float(self.params.get("accumulation_time_ms", 2.0))
        step = acc_ms * 1000.0
    
        if self.display_t0 is None:
            self.display_t0 = float(np.min(ts))
    
        new_t0 = float(self.display_t0) + direction * step

        data_min = float(np.min(ts))
        data_max = float(np.max(ts))
        if new_t0 < data_min:
            new_t0 = data_min
        if new_t0 + step > data_max:
            new_t0 = data_max - step
    
        self.display_t0 = new_t0
        self.update_pseudoframe_from_data("Display window shifted.")

    def render_plot_to_texture(self, fig, texture_tag: str) -> None:  
        try:
            dpg.set_value(texture_tag, figure_to_rgba_flat(fig))
        except Exception as exc:
            self.log_status(f"Error rendering plot: {exc}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_status(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        if dpg.does_item_exist("status_log_text"):
            current_text = dpg.get_value("status_log_text")
            new_text = f"[{timestamp}] {message}\n{current_text}"
            dpg.set_value("status_log_text", new_text)

    # ------------------------------------------------------------------
    # Export / Import filtered events
    # ------------------------------------------------------------------
    def browse_save_filtered_file(self, sender=None, app_data=None, user_data=None):
        default_path = None
        try:
            if self.data_file:
                default_path = str(Path(self.data_file).resolve().parent)
        except Exception:
            default_path = None
        if default_path and dpg.does_item_exist("save_filtered_file_dialog"):
            dpg.configure_item("save_filtered_file_dialog", default_path=default_path)
        dpg.show_item("save_filtered_file_dialog")

    def on_save_filtered_file_selected(self, sender, app_data) -> None:
        file_path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
        if not file_path:
            dpg.set_value("save_filtered_status", "No file selected.")
            return
        file_path = str(file_path).strip().strip('"').strip("'")
        dpg.set_value("save_filtered_file_input", file_path)
        ext = Path(file_path).suffix.lower()
        if ext in (".npz", ".mat") and dpg.does_item_exist("save_filtered_format"):
            dpg.set_value("save_filtered_format", ext)
        dpg.set_value("save_filtered_status", f"Target set: {Path(file_path).name}")

    def on_save_filtered_path_typed(self, sender, app_data) -> None:
        path = (app_data or "").strip().strip('"').strip("'")
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext in (".npz", ".mat") and dpg.does_item_exist("save_filtered_format"):
            dpg.set_value("save_filtered_format", ext)

    def save_filtered_events(self, sender=None, app_data=None, user_data=None) -> None:
        if not getattr(self, "raw_events", None):
            dpg.set_value("save_filtered_status", "No data loaded.")
            return

        self.apply_event_filters()
        if not getattr(self, "filtered_events", None):
            dpg.set_value("save_filtered_status", "No filtered events available.")
            return

        X, Y, T, P = self.filtered_events
        target = ""
        if dpg.does_item_exist("save_filtered_file_input"):
            target = str(dpg.get_value("save_filtered_file_input") or "").strip()
        fmt = ".npz"
        if dpg.does_item_exist("save_filtered_format"):
            fmt = str(dpg.get_value("save_filtered_format") or ".npz").strip()

        if not target:
            base_dir = Path(self.data_file).resolve().parent if self.data_file else Path.cwd()
            base_name = Path(self.data_file).stem if self.data_file else "events"
            target = str(base_dir / f"{base_name}_filtered{fmt}")

        target_path = Path(target)
        if target_path.suffix.lower() not in (".npz", ".mat"):
            target_path = target_path.with_suffix(fmt)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.suffix.lower() == ".npz":
                payload = {"X": np.asarray(X), "Y": np.asarray(Y), "T": np.asarray(T)}
                if P is not None:
                    payload["P"] = np.asarray(P)
                np.savez_compressed(str(target_path), **payload, width=int(self.params.get("width", 0)), height=int(self.params.get("height", 0)))
            else:
                from scipy.io import savemat  
                payload = {"X": np.asarray(X), "Y": np.asarray(Y), "T": np.asarray(T)}
                if P is not None:
                    payload["P"] = np.asarray(P)
                payload["width"] = int(self.params.get("width", 0))
                payload["height"] = int(self.params.get("height", 0))
                savemat(str(target_path), payload)
            dpg.set_value("save_filtered_status", f"Saved: {target_path.name} ({int(np.asarray(X).size):,} events)")
            self.log_status(f"Filtered events saved to: {target_path}")
        except Exception as exc:
            dpg.set_value("save_filtered_status", f"Save failed: {exc}")
            self.log_status(f"Save filtered events failed: {exc}")