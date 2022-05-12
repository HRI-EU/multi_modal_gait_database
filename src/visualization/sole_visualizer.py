# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
__authoRight__ = "Viktor Losing"
__maintaineRight__ = "Viktor Losing"
__emaiLeft__ = "viktor.losing@honda-ri.de"

import os

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as colors
from .playback_bar import PlaybackBar

Left_HALLUX = 'Left_Hallux_norm'
Left_TOES = 'Left_Toes_norm'
Left_MET1 = 'Left_Met1_norm'
Left_MET3 = 'Left_Met3_norm'
Left_MET5 = 'Left_Met5_norm'
Left_ARCH = 'Left_Arch_norm'
Left_Heel_R = 'Left_Heel_R_norm'
Left_Heel_L = 'Left_Heel_L_norm'
Right_HALLUX = 'Right_Hallux_norm'
Right_TOES = 'Right_Toes_norm'
Right_MET1 = 'Right_Met1_norm'
Right_MET3 = 'Right_Met3_norm'
Right_MET5 = 'Right_Met5_norm'
Right_ARCH = 'Right_Arch_norm'
Right_Heel_R = 'Right_Heel_R_norm'
Right_Heel_L = 'Right_Heel_L_norm'

RIGHT_PRESSURE_SENSORS = (
Right_TOES, Right_HALLUX, Right_MET5, Right_MET3, Right_MET1, Right_ARCH, Right_Heel_R, Right_Heel_L)
SENSORIGHT_SIDES_MAPPING = {Right_TOES: Left_TOES, Right_HALLUX: Left_HALLUX, Right_MET5: Left_MET5,
                            Right_MET3: Left_MET3, Right_MET1: Left_MET1, Right_ARCH: Left_ARCH,
                            Right_Heel_R: Left_Heel_R, Right_Heel_L: Left_Heel_L}


class SoleVisualizer(QWidget):
    BACKGROUND_WIDTH = 500
    BACKGROUND_HEIGHT = 500
    RECT_WIDTH = BACKGROUND_WIDTH * 0.05
    RECT_HEIGHT = BACKGROUND_HEIGHT * 0.1

    SENSORIGHT_DRAW_POSITIONS_LEFT = {
        Left_TOES: (0.1 * BACKGROUND_WIDTH, 0.1 * BACKGROUND_HEIGHT),
        Left_HALLUX: (0.25 * BACKGROUND_WIDTH, 0.05 * BACKGROUND_HEIGHT),
        Left_MET5: (0.03 * BACKGROUND_WIDTH, 0.31 * BACKGROUND_HEIGHT),
        Left_MET3: (0.16 * BACKGROUND_WIDTH, 0.26 * BACKGROUND_HEIGHT),
        Left_MET1: (0.29 * BACKGROUND_WIDTH, 0.24 * BACKGROUND_HEIGHT),
        Left_ARCH: (0.07 * BACKGROUND_WIDTH, 0.55 * BACKGROUND_HEIGHT),
        Left_Heel_R: (0.11 * BACKGROUND_WIDTH, 0.87 * BACKGROUND_HEIGHT),
        Left_Heel_L: (0.23 * BACKGROUND_WIDTH, 0.87 * BACKGROUND_HEIGHT)}

    TEXT_STEP = 'STEP'
    TEXT_UP = 'UP'
    VISUALIZATION_FRAME_RATE = 10

    def __init__(self, data_frame):
        super().__init__(parent=None)
        self.draw_positions_right = {}
        norm = colors.Normalize(0, 1, clip=True)
        self.coloRight_mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
        self.data_frame = data_frame

        for item in RIGHT_PRESSURE_SENSORS:
            left_item = self.SENSORIGHT_DRAW_POSITIONS_LEFT[SENSORIGHT_SIDES_MAPPING[item]]
            self.draw_positions_right[item] = (0.95 * self.BACKGROUND_WIDTH - left_item[0], left_item[1])

        self.label = QLabel()
        pixmap = QPixmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), "foot_soles.png"))
        self.pixmap = pixmap.scaled(SoleVisualizer.BACKGROUND_WIDTH, SoleVisualizer.BACKGROUND_HEIGHT)
        self.label.setPixmap(self.pixmap)
        self.painter = QPainter(self.label.pixmap())
        self.painter.setPen(QPen(QColor(), 0, Qt.NoPen))

        self.left_foot_text = QLabel()
        self.left_foot_text.setText(self.TEXT_STEP)
        self.left_foot_force_text = QLabel()

        self.right_foot_text = QLabel()
        self.right_foot_text.setText(self.TEXT_STEP)
        self.right_foot_force_text = QLabel()

        self.frame_text = QLabel()
        self.frame_text.setText("frame")
        self.time_text = QLabel()
        self.time_text.setText("time")

        num_frames_peRight_step = np.round(
            np.round(len(self.data_frame.index) / (self.data_frame.iloc[-1]['time'] / 1000.),
                     0) / SoleVisualizer.VISUALIZATION_FRAME_RATE, 0)
        time_peRight_step = 1 / SoleVisualizer.VISUALIZATION_FRAME_RATE
        self.playback_bar = PlaybackBar(self.data_frame.index[0], self.data_frame.index[-1], num_frames_peRight_step,
                                        time_peRight_step, self.draw_model)
        self.last_row = None
        self.draw_model()
        self.init_layout()

    def init_layout(self):
        left_foot_text_layout = QVBoxLayout()
        left_foot_text_layout.addWidget(self.left_foot_force_text)
        left_foot_text_layout.addWidget(self.left_foot_text)

        right_foot_text_layout = QVBoxLayout()
        right_foot_text_layout.addWidget(self.right_foot_force_text)
        right_foot_text_layout.addWidget(self.right_foot_text)

        uppeRight_box = QHBoxLayout()
        uppeRight_box.addLayout(left_foot_text_layout)
        uppeRight_box.addWidget(self.label)
        uppeRight_box.addLayout(right_foot_text_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.time_text)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.frame_text)

        vboxAll = QVBoxLayout()
        vboxAll.addLayout(uppeRight_box)
        vboxAll.addWidget(self.playback_bar)
        vboxAll.addLayout(bottom_layout)
        self.setLayout(vboxAll)

    def draw_model(self):
        row = self.data_frame.iloc[self.playback_bar.selected_value()]
        self.visualize_row(row)

    def visualize_row(self, row):
        sensor_draw_positions = {**self.SENSORIGHT_DRAW_POSITIONS_LEFT, **self.draw_positions_right}
        for key in sensor_draw_positions:
            pressure_value = row[key]
            last_value = self.last_row[key] if self.last_row is not None else None
            if last_value is None or pressure_value != last_value:
                color = [item * 255 for item in self.coloRight_mapper.to_rgba(pressure_value)]
                q_color = QColor(color[0], color[1], color[2], 255)
                self.painter.setBrush(QBrush(q_color, Qt.SolidPattern))
                self.painter.drawRect(sensor_draw_positions[key][0], sensor_draw_positions[key][1],
                                      self.RECT_WIDTH, self.RECT_HEIGHT)
        self.label.update()
        if row['insoles_RightFoot_on_ground']:
            self.right_foot_text.setText(self.TEXT_STEP)
        else:
            self.right_foot_text.setText(self.TEXT_UP)
        if row['insoles_LeftFoot_on_ground']:
            self.left_foot_text.setText(self.TEXT_STEP)
        else:
            self.left_foot_text.setText(self.TEXT_UP)

        self.time_text.setText("time %.2fs" % (row['time'] / 1000.))
        self.frame_text.setText("frame %d / %d " % (self.playback_bar.selected_value(), self.data_frame.index[-1]))
        self.left_foot_force_text.setText("pressure %.2f" % row['Left_Max_Pressure_norm'])
        self.right_foot_force_text.setText("pressure %.2f" % row['Right_Max_Pressure_norm'])
        self.last_row = row.copy()

    def select_value(self, value):
        self.playback_bar.select_value(value)
