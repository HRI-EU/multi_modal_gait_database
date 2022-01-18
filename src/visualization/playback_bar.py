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

import threading

from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QScrollBar
from PyQt5 import QtCore


class RepeatedTimer(object):
    def __init__(self, interval, function, on_start_function, on_stop_function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.on_stop_function = on_stop_function
        self.on_start_function = on_start_function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False

    def _run(self):
        self.is_running = False
        self.start()
        if not self.function(*self.args, **self.kwargs):
            self.stop()

    def start(self):
        if not self.is_running:
            self._timer = threading.Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True
            self.on_start_function()

    def stop(self):
        if self.is_running:
            self._timer.cancel()
            self.is_running = False
            self.on_stop_function()


class PlaybackBar(QWidget):
    def __init__(self, slider_minimum, slider_maximum, num_frames_per_step, time_per_step, on_slider_change_event):
        super().__init__(parent=None)
        self.num_frames_per_step = num_frames_per_step
        self.slider = QScrollBar()
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setMinimum(slider_minimum)
        self.slider.setMaximum(slider_maximum)

        self.button_far_back = QPushButton('<<<')
        self.button_far_back.setFixedWidth(25)
        self.button_far_back.clicked.connect(self.button_far_back_click)

        self.button_med_back = QPushButton('<<')
        self.button_med_back.setFixedWidth(25)
        self.button_med_back.clicked.connect(self.button_med_back_click)

        self.button_back = QPushButton('<')
        self.button_back.setFixedWidth(25)
        self.button_back.clicked.connect(self.button_back_click)

        self.button_forward = QPushButton('>')
        self.button_forward.setFixedWidth(25)
        self.button_forward.clicked.connect(self.button_forward_click)

        self.button_med_forward = QPushButton('>>')
        self.button_med_forward.setFixedWidth(25)
        self.button_med_forward.clicked.connect(self.button_med_forward_click)

        self.button_far_forward = QPushButton('>>>')
        self.button_far_forward.setFixedWidth(25)
        self.button_far_forward.clicked.connect(self.button_far_forward_click)

        self.button_play = QPushButton('Play')
        self.button_play.setFixedWidth(80)
        self.button_play.clicked.connect(self.button_play_click)


        slider_button_layout = QHBoxLayout(self)
        slider_button_layout.addWidget(self.button_far_back)
        slider_button_layout.addWidget(self.button_med_back)
        slider_button_layout.addWidget(self.button_back)
        slider_button_layout.addWidget(self.slider)
        slider_button_layout.addWidget(self.button_forward)
        slider_button_layout.addWidget(self.button_med_forward)
        slider_button_layout.addWidget(self.button_far_forward)
        slider_button_layout.addWidget(self.button_play)

        self.play_timer = RepeatedTimer(time_per_step, self.one_step_forward, self.update_button_play, self.update_button_play)
        self.slider.valueChanged.connect(lambda: on_slider_change_event())
        self.slider.setValue(slider_minimum)

    def selected_value(self):
        return self.slider.value()

    def select_value(self, value):
        return self.slider.setValue(value)

    def set_max_value(self, value):
        return self.slider.setMaximum(value)

    def button_forward_click(self):
        value = min(self.slider.value() + 1, self.slider.maximum())
        self.slider.setValue(value)

    def one_step_forward(self):
        value = min(self.slider.value() + self.num_frames_per_step, self.slider.maximum())
        self.slider.setValue(value)
        return self.slider.value() < self.slider.maximum()

    def button_med_forward_click(self):
        value = min(self.slider.value() + self.num_frames_per_step, self.slider.maximum())
        self.slider.setValue(value)

    def button_far_forward_click(self):
        value = min(self.slider.value() + self.num_frames_per_step * 5, self.slider.maximum())
        self.slider.setValue(value)

    def button_back_click(self):
        value = max(self.slider.value() - 1, self.slider.minimum())
        self.slider.setValue(value)

    def button_med_back_click(self):
        value = max(self.slider.value() - self.num_frames_per_step, self.slider.minimum())
        self.slider.setValue(value)

    def button_far_back_click(self):
        value = max(self.slider.value() - self.num_frames_per_step * 5, self.slider.minimum())
        self.slider.setValue(value)

    def update_button_play(self):
        if self.play_timer.is_running:
            self.button_play.setText('Stop')
        else:
            self.button_play.setText('Play')

    def button_play_click(self):
        if self.play_timer.is_running:
            self.play_timer.stop()
        else:
            self.play_timer.start()
        self.update_button_play()
