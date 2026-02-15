from __future__ import annotations

import dearpygui.dearpygui as dpg

from config.constants import APP_TITLE
from interface import PreprocessingTab, DetectionTab, TrackingTab, ValidationTab
from core.layout import LayoutManager


class EVTrackApp:
    def __init__(self) -> None:
        self.preprocessing_tab = PreprocessingTab()
        self.detection_tab = DetectionTab(self.preprocessing_tab)
        self.tracking_tab = TrackingTab(self.preprocessing_tab, self.detection_tab)
        self.validation_tab = ValidationTab(self.preprocessing_tab, self.tracking_tab)
        self.layout = LayoutManager(
            [
                self.preprocessing_tab,
                self.detection_tab,
                self.tracking_tab,
                self.validation_tab,
                
            ]
        )

    def start(self) -> None:
        self.layout.build()
        while dpg.is_dearpygui_running():
            self.detection_tab.process_background_tasks()
            self.tracking_tab.process_background_tasks()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def run() -> None:
    print("=" * 60)
    print(APP_TITLE)
    print("=" * 60)
    app = EVTrackApp()
    app.start()

