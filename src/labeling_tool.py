# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
__author__ = 'Viktor Losing'
__maintainer__ = 'Viktor Losing'
__email__ = 'viktor.losing@honda-ri.de'

import sys
import argparse
import os

import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, \
    QTableWidget, QAbstractItemView, QTableWidgetItem, QApplication
import numpy as np
import common
from visualization.sole_visualizer import SoleVisualizer
from visualization.xsens_playback_tool import XsensPlaybackTool
from visualization.playback_bar import PlaybackBar
from visualization.eyetracker_visualizer import EyeTrackerVisualizer

IMU_PREFIX = 'imu'
INSOLES_PREFIX = 'insoles'
EYE_TRACKER_PREFIX = 'eyetracker'


class LabelingTool(QWidget):
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 500
    VISUALIZATION_FRAME_RATE = 10
    LBL_WALK = 'walk'
    LBL_STAIRS_UP = 'stairs_up'
    LBL_STAIRS_DOWN = 'stairs_down'
    LBL_SLOPE_UP = 'slope_up'
    LBL_SLOPE_DOWN = 'slope_down'
    LBL_PAVEMENT_UP = 'pavement_up'
    LBL_PAVEMENT_DOWN = 'pavement_down'
    LBL_CURVE_LEFT = 'curve_left'
    LBL_CURVE_RIGHT = 'curve_right'
    LBL_TURN_LEFT = 'turn_left'
    LBL_TURN_RIGHT = 'turn_right'

    def __init__(self, source_path, course, subject_id, labels_path=None):
        super().__init__(parent=None)
        self.subject_path = common.get_path_to_course_subject(source_path, course, subject_id)
        eye_tracker_data_frame = pd.read_csv(os.path.join(self.subject_path, common.EYE_TRACKER_FILE_NAME))
        insoles_data_frame = pd.read_csv(os.path.join(self.subject_path, common.INSOLES_FILE_NAME))
        imu_data_frame = pd.read_csv(os.path.join(self.subject_path, common.IMU_FILE_NAME))
        if labels_path is None:
            self.labels_data_frame = pd.read_csv(os.path.join(self.subject_path, common.LABELS_FILE_NAME))
        else:
            self.labels_data_frame = pd.read_csv(labels_path)
        video_path = os.path.join(self.subject_path, common.VIDEO_FILE_NAME)

        insoles_data_frame['insoles_RightFoot_on_ground'] = self.labels_data_frame['insoles_RightFoot_on_ground']
        insoles_data_frame['insoles_LeftFoot_on_ground'] = self.labels_data_frame['insoles_LeftFoot_on_ground']

        self.sole_visualizer = SoleVisualizer(insoles_data_frame)
        self.xsens_playback_tool = XsensPlaybackTool(imu_data_frame)
        self.eye_tracker_visualizer = EyeTrackerVisualizer(eye_tracker_data_frame, video_path)

        self.resize(LabelingTool.WINDOW_WIDTH, LabelingTool.WINDOW_HEIGHT)
        self.stream_frame_rate = 60

        self.frame_text = QLabel()
        self.frame_text.setText('frame')
        self.time_text = QLabel()
        self.time_text.setText('time')

        self.label_text = QLabel()
        self.label_text.setText('label')

        self.combobox_label = QComboBox(self)
        self.combobox_label.addItem(LabelingTool.LBL_WALK)
        self.combobox_label.addItem(LabelingTool.LBL_STAIRS_UP)
        self.combobox_label.addItem(LabelingTool.LBL_STAIRS_DOWN)
        self.combobox_label.addItem(LabelingTool.LBL_SLOPE_UP)
        self.combobox_label.addItem(LabelingTool.LBL_SLOPE_DOWN)
        self.combobox_label.addItem(LabelingTool.LBL_PAVEMENT_UP)
        self.combobox_label.addItem(LabelingTool.LBL_PAVEMENT_DOWN)
        self.combobox_label.addItem(LabelingTool.LBL_CURVE_LEFT)
        self.combobox_label.addItem(LabelingTool.LBL_CURVE_RIGHT)
        self.combobox_label.addItem(LabelingTool.LBL_TURN_LEFT)
        self.combobox_label.addItem(LabelingTool.LBL_TURN_RIGHT)

        self.button_export = QPushButton('Export')
        # self.button_export.setFixedWidth(80)
        self.button_export.clicked.connect(self.button_export_click)

        self.button_add = QPushButton('Add')
        self.button_add.setFixedWidth(80)
        self.button_add.clicked.connect(self.button_add_click)

        self.button_remove = QPushButton('Remove')
        self.button_remove.setFixedWidth(80)
        self.button_remove.clicked.connect(self.button_remove_click)

        self.table_labels = QTableWidget()
        self.table_labels.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_labels.setWindowTitle('Label')
        self.table_labels.horizontalHeader().setStretchLastSection(True)
        self.table_labels.setRowCount(0)
        self.table_labels.setFixedHeight(150)
        self.table_labels.setColumnCount(2)
        self.table_labels.setHorizontalHeaderLabels(('Frame', 'Label'))
        self.table_labels.setColumnWidth(1, 30)
        self.table_labels.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_labels.doubleClicked.connect(self.table_double_click)

        num_frames_per_step = np.round(
            np.round(len(self.labels_data_frame.index) / (self.labels_data_frame.iloc[-1]['time'] / 1000.),
                     0) / LabelingTool.VISUALIZATION_FRAME_RATE, 0)
        time_per_step = 1 / LabelingTool.VISUALIZATION_FRAME_RATE
        self.playback_bar = PlaybackBar(0, self.labels_data_frame.index[-1], num_frames_per_step, time_per_step,
                                        self.draw_model)
        self.init_layout()
        self.init_labels()

    def button_export_click(self):
        save_filepath, _ = QFileDialog.getSaveFileName(self, 'Save labeled dataframe', self.subject_path,
                                                       'csv files (*.csv)', options=QFileDialog.DontUseNativeDialog)
        if save_filepath != '':
            labels = []
            last_index = 0
            last_label = LabelingTool.LBL_WALK
            for i in range(self.table_labels.rowCount()):
                if self.table_labels.item(i, 0) is None or self.table_labels.item(i, 0).text() == '':
                    break
                else:
                    labels += (int(self.table_labels.item(i, 0).text()) - last_index) * [last_label]
                    last_index = int(self.table_labels.item(i, 0).text())
                    last_label = self.table_labels.item(i, 1).text()
            labels += (self.labels_data_frame.index[-1] + 1 - last_index) * [last_label]
            self.labels_data_frame['walk_mode'] = labels
            self.labels_data_frame.to_csv(save_filepath)
            os.chmod(save_filepath, 0o660)
            print('saved labeled data frame %s' % save_filepath)

    def table_double_click(self, mi):
        row_idx = mi.row()
        self.playback_bar.select_value(int(self.table_labels.item(row_idx, 0).text()))

    def button_remove_click(self):
        indexes = self.table_labels.selectionModel().selectedRows()
        for index in sorted(indexes):
            self.table_labels.removeRow(index.row())

    def button_add_click(self):
        row_count = self.table_labels.rowCount()  # necessary even when there are no rows in the table
        current_frame = self.playback_bar.selected_value()
        target_row_index = 0
        for i in range(row_count):
            if self.table_labels.item(i, 0) is None or self.table_labels.item(i, 0).text() != '' and int(
                    self.table_labels.item(i, 0).text()) < current_frame:
                target_row_index = i + 1
        self.table_labels.insertRow(target_row_index)
        self.table_labels.setItem(target_row_index, 0, QTableWidgetItem(str(current_frame)))
        self.table_labels.setItem(target_row_index, 1, QTableWidgetItem(self.combobox_label.currentText()))

    def init_labels(self):
        labels = self.labels_data_frame['walk_mode']
        last_label = ''
        for idx, label in enumerate(labels):
            if label != last_label:
                row_idx = self.table_labels.rowCount()
                self.table_labels.insertRow(row_idx)
                self.table_labels.setItem(row_idx, 0, QTableWidgetItem(str(idx)))
                self.table_labels.setItem(row_idx, 1, QTableWidgetItem(label))
                last_label = label

    def get_current_label(self):
        frame_idx = self.playback_bar.selected_value()
        last_label = self.LBL_WALK
        last_index = 0
        for i in range(self.table_labels.rowCount()):
            if self.table_labels.item(i, 0) is None or self.table_labels.item(i, 0).text() == '':
                break
            else:
                current_index = int(self.table_labels.item(i, 0).text())
                if last_index <= frame_idx < current_index:
                    return last_label
                last_label = self.table_labels.item(i, 1).text()
        return last_label

    def init_layout(self):
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.xsens_playback_tool)
        top_layout.addWidget(self.sole_visualizer)
        top_layout.addWidget(self.eye_tracker_visualizer)

        time_layout = QHBoxLayout()
        time_layout.addWidget(self.time_text)
        time_layout.addWidget(self.frame_text)
        time_layout.addWidget(self.label_text)

        bottom_left_vbox = QVBoxLayout()
        bottom_left_vbox.addWidget(self.combobox_label)
        bottom_left_vbox.addWidget(self.button_export)
        bottom_left_vbox.addStretch(1)

        bottom_right_vbox = QVBoxLayout()
        bottom_right_vbox.addWidget(self.button_add)
        bottom_right_vbox.addWidget(self.button_remove)
        bottom_right_vbox.addStretch(1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(bottom_left_vbox)
        bottom_layout.addLayout(bottom_right_vbox)
        bottom_layout.addWidget(self.table_labels)

        root_layout = QVBoxLayout()
        root_layout.addLayout(top_layout)
        root_layout.addWidget(self.playback_bar)
        root_layout.addLayout(time_layout)
        root_layout.addLayout(bottom_layout)
        self.setLayout(root_layout)

    def draw_model(self):
        self.xsens_playback_tool.select_value(self.playback_bar.selected_value())
        self.sole_visualizer.select_value(self.playback_bar.selected_value())
        self.eye_tracker_visualizer.select_value(self.playback_bar.selected_value())
        self.time_text.setText('time %.2fs' % (self.playback_bar.selected_value() / float(self.stream_frame_rate)))
        self.frame_text.setText(
            'frame %d / %d ' % (self.playback_bar.selected_value(), self.playback_bar.slider.maximum()))
        self.label_text.setText('label %s' % (self.get_current_label()))


if __name__ == '__main__':
    import logging

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    argparser = argparse.ArgumentParser(description='Label synchronized stream.')
    argparser.add_argument('--source_path', '-p', action='store', required=True,
                           dest='source_path', type=str,
                           help='path to dataset folder')

    argparser.add_argument('--labels_path', '-l', action='store', required=False, default=None,
                           dest='labels_path', type=str,
                           help='path to labels file name')

    argparser.add_argument('--course', '-c', action='store', required=True, default=common.COURSE_A,
                           dest='course', type=str,
                           help='specify the walking course (%s, %s, %s)' % (
                           common.COURSE_A, common.COURSE_B, common.COURSE_C))

    argparser.add_argument('--subject', '-s', action='store', required=True, default=1,
                           dest='subject_id', type=int,
                           help='specify subject via id')

    app = QApplication(sys.argv)
    args = argparser.parse_args()
    w = LabelingTool(args.source_path, args.course, args.subject_id, args.labels_path)

    w.show()
    sys.exit(app.exec_())
