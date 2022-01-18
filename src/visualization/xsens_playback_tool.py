# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
__author__ = "Viktor Losing"
__maintainer__ = "Viktor Losing"
__email__ = "viktor.losing@honda-ri.de"

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

from .playback_bar import PlaybackBar
from .xsens_visualizer import XSensVisualizer


class XsensPlaybackTool(QWidget):
    WINDOW_WIDTH = 500
    WINDOW_HEIGHT = 500
    VISUALIZATION_FRAME_RATE = 10

    def __init__(self, data_frame):
        super().__init__(parent=None)
        self.data_frame = data_frame
        self.resize(XsensPlaybackTool.WINDOW_WIDTH, XsensPlaybackTool.WINDOW_HEIGHT)
        self.xsens_visualizer = XSensVisualizer()

        self.frame_text = QLabel()
        self.frame_text.setText("frame")
        self.time_text = QLabel()
        self.time_text.setText("time")
        num_frames_per_step = np.round(len(self.data_frame.index) / (self.data_frame.iloc[-1]['time'] / 1000.) / XsensPlaybackTool.VISUALIZATION_FRAME_RATE, 0)
        time_per_step = 1 / XsensPlaybackTool.VISUALIZATION_FRAME_RATE
        self.playback_bar = PlaybackBar(self.data_frame.index[0], self.data_frame.index[-1], num_frames_per_step, time_per_step, self.update_current_frame)
        self.init_layout()
        self.update_current_frame()

    def init_layout(self):
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.time_text)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.frame_text)

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.xsens_visualizer)
        root_layout.addWidget(self.playback_bar)
        root_layout.addLayout(bottom_layout)
        self.setLayout(root_layout)

    def select_value(self, value):
        self.playback_bar.select_value(value)

    def update_current_frame(self):
        data_row = self.data_frame.iloc[self.playback_bar.selected_value()]
        self.xsens_visualizer.update_model(data_row)
        self.time_text.setText("time %.2fs" % (data_row['time']/1000.))
        self.frame_text.setText("frame %d / %d " % (self.playback_bar.selected_value(), self.data_frame.index[-1]))
