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

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2
import logging

from .playback_bar import PlaybackBar


class EyeTrackerVisualizer(QWidget):
    BACKGROUND_WIDTH = 500
    BACKGROUND_HEIGHT = 600
    RECT_WIDTH = BACKGROUND_WIDTH * 0.05
    RECT_HEIGHT = BACKGROUND_HEIGHT * 0.1
    VISUALIZATION_FRAME_RATE = 10

    def __init__(self, data_frame, video_path, image_map=None):
        super().__init__(parent=None)
        self.data_frame = data_frame
        self.video_path = video_path
        self.current_image_index = -1
        self.current_image = None
        self.video_capture = None
        self.image_map = image_map
        if self.image_map is None:
            self.image_map = self.generate_image_map()

        self.label = QLabel()

        self.frame_text = QLabel()
        self.frame_text.setText("frame")

        self.index_text = QLabel()
        self.index_text.setText("index")
        self.time_text = QLabel()
        self.time_text.setText("time")

        num_frames_per_step = np.round(np.round(len(self.data_frame.index) / (self.data_frame.iloc[-1]['gaze_timestamp']), 0) / EyeTrackerVisualizer.VISUALIZATION_FRAME_RATE, 0)
        time_per_step = 1 / EyeTrackerVisualizer.VISUALIZATION_FRAME_RATE
        self.playback_bar = PlaybackBar(self.data_frame.index[0], self.data_frame.index[-1], num_frames_per_step, time_per_step, self.draw_model)
        self.draw_model()
        self.init_layout()

    def init_layout(self):
        upper_box = QHBoxLayout()
        upper_box.addWidget(self.label)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.time_text)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.frame_text)
        bottom_layout.addWidget(self.index_text)

        vboxAll = QVBoxLayout()
        vboxAll.addLayout(upper_box)
        vboxAll.addWidget(self.playback_bar)
        vboxAll.addLayout(bottom_layout)
        self.setLayout(vboxAll)

    def generate_image_map(self):
        logging.info("loading video...")
        image_indices = self.data_frame["world_index"].unique().astype(np.int).tolist()
        video_capture = cv2.VideoCapture(self.video_path)
        current_image_index = 0
        image_map = {}
        success, image = video_capture.read()
        while success:
            if current_image_index in image_indices:
                image = cv2.resize(image, (self.BACKGROUND_WIDTH, self.BACKGROUND_HEIGHT-100))
                image_map[current_image_index] = image
            success, image = video_capture.read()
            current_image_index += 1
        logging.info("video loaded")
        return image_map

    def draw_model(self):
        row = self.data_frame.iloc[self.playback_bar.selected_value()]
        self.visualize_row(row)

    def visualize_row(self, row):
        #image = self.get_image(image_index)
        image = self.image_map[int(row["world_index"])]
        pixmap = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped())
        self.label.setPixmap(pixmap)
        self.label.update()
        self.time_text.setText("time %.2fs" % (row['gaze_timestamp']))
        self.frame_text.setText("frame %d / %d " % (self.playback_bar.selected_value(), self.data_frame.index[-1]))
        self.index_text.setText("index %d" % (int(row["world_index"])))

    def select_value(self, value):
        self.playback_bar.select_value(value)
