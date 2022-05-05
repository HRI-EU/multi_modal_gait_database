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
    QTableWidget, QAbstractItemView, QTableWidgetItem, QApplication, QTabWidget
from PyQt5 import QtCore

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

    LBL_TYPE_MODE = "mode"
    LBL_TYPE_ORIENTATION = "orientation"
    LBL_TYPE_INTERACTION = "interaction"

    LBL_COLUMN_MODE = "walk_mode"
    LBL_COLUMN_ORIENTATION = "walk_orientation"
    LBL_COLUMN_INTERACTION = "walk_interaction"

    LBL_WALK = 'walk'
    LBL_STAIRS_UP = 'stairs_up'
    LBL_STAIRS_DOWN = 'stairs_down'
    LBL_SLOPE_UP = 'slope_up'
    LBL_SLOPE_DOWN = 'slope_down'
    LBL_PAVEMENT_UP = 'pavement_up'
    LBL_PAVEMENT_DOWN = 'pavement_down'
    LBL_STRAIGHT = 'straight'
    LBL_CURVE_LEFT = 'curve_left'
    LBL_CURVE_RIGHT = 'curve_right'
    LBL_TURN_CLOCKWISE = 'turn_around_clockwise'
    LBL_TURN_COUNTER_CLOCKWISE = 'turn_around_counter_clockwise'
    LBL_YES = 'yes'
    LBL_NO = 'no'

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
        #self.combobox_label.addItem("UP")
        #self.combobox_label.addItem("DOWN")
        self.combobox_label.addItem(LabelingTool.LBL_WALK)
        self.combobox_label.addItem(LabelingTool.LBL_STAIRS_UP)
        self.combobox_label.addItem(LabelingTool.LBL_STAIRS_DOWN)
        self.combobox_label.addItem(LabelingTool.LBL_SLOPE_UP)
        self.combobox_label.addItem(LabelingTool.LBL_SLOPE_DOWN)
        self.combobox_label.addItem(LabelingTool.LBL_PAVEMENT_UP)
        self.combobox_label.addItem(LabelingTool.LBL_PAVEMENT_DOWN)

        self.button_export = QPushButton('Export')
        # self.button_export.setFixedWidth(80)
        self.button_export.clicked.connect(self.button_export_click)

        self.button_add = QPushButton('Add')
        self.button_add.setFixedWidth(80)
        self.button_add.clicked.connect(self.button_add_click)

        self.button_remove = QPushButton('Remove')
        self.button_remove.setFixedWidth(80)
        self.button_remove.clicked.connect(self.button_remove_click)

        self.table_mode = self.new_label_tabel()
        self.table_orientation = self.new_label_tabel()
        self.table_interaction = self.new_label_tabel()

        self.tab_sheet_labels = QTabWidget()
        self.tab_sheet_labels.addTab(self.table_mode, self.LBL_TYPE_MODE)
        self.tab_sheet_labels.addTab(self.table_orientation, self.LBL_TYPE_ORIENTATION)
        self.tab_sheet_labels.addTab(self.table_interaction, self.LBL_TYPE_INTERACTION)
        self.tab_sheet_labels.currentChanged.connect(self.on_tab_sheet_changed)



        num_frames_per_step = np.round(
            np.round(len(self.labels_data_frame.index) / (self.labels_data_frame.iloc[-1]['time'] / 1000.),
                     0) / LabelingTool.VISUALIZATION_FRAME_RATE, 0)
        time_per_step = 1 / LabelingTool.VISUALIZATION_FRAME_RATE
        self.playback_bar = PlaybackBar(0, self.labels_data_frame.index[-1], num_frames_per_step, time_per_step,
                                        self.draw_model)
        self.init_layout()
        self.init_labels()

    def new_label_tabel(self):
        table = QTableWidget()
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setWindowTitle('Label')
        table.horizontalHeader().setStretchLastSection(True)
        table.setRowCount(0)
        table.setFixedHeight(150)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(('Frame', 'Label'))
        table.setColumnWidth(1, 30)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.doubleClicked.connect(self.table_double_click)
        return table

    def on_tab_sheet_changed(self):
        self.label_text.setText('label %s' % (self.get_current_label()))
        if self.tab_sheet_labels.currentIndex() == 0:
            self.combobox_label.clear()
            self.combobox_label.addItem(LabelingTool.LBL_WALK)
            self.combobox_label.addItem(LabelingTool.LBL_STAIRS_UP)
            self.combobox_label.addItem(LabelingTool.LBL_STAIRS_DOWN)
            self.combobox_label.addItem(LabelingTool.LBL_SLOPE_UP)
            self.combobox_label.addItem(LabelingTool.LBL_SLOPE_DOWN)
            self.combobox_label.addItem(LabelingTool.LBL_PAVEMENT_UP)
            self.combobox_label.addItem(LabelingTool.LBL_PAVEMENT_DOWN)
        elif self.tab_sheet_labels.currentIndex() == 1:
            self.combobox_label.clear()
            self.combobox_label.addItem(LabelingTool.LBL_STRAIGHT)
            self.combobox_label.addItem(LabelingTool.LBL_CURVE_RIGHT)
            self.combobox_label.addItem(LabelingTool.LBL_CURVE_LEFT)
            self.combobox_label.addItem(LabelingTool.LBL_TURN_CLOCKWISE)
            self.combobox_label.addItem(LabelingTool.LBL_TURN_COUNTER_CLOCKWISE)
        elif self.tab_sheet_labels.currentIndex() == 2:
            self.combobox_label.clear()
            self.combobox_label.addItem(LabelingTool.LBL_NO)
            self.combobox_label.addItem(LabelingTool.LBL_YES)

    def keyPressEvent(self, event):
        #if event.key() == QtCore.Qt.Key_W:
        #    self.button_add_click("UP")
        #elif event.key() == QtCore.Qt.Key_S:
        #    self.button_add_click("DOWN")
        if event.key() == QtCore.Qt.Key_D:
            self.playback_bar.button_forward_click()
            self.draw_model()
        elif event.key() == QtCore.Qt.Key_A:
            self.playback_bar.button_back_click()
            self.draw_model()

    def get_current_table(self):
        if self.tab_sheet_labels.currentIndex() == 0:
            return self.table_mode
        elif self.tab_sheet_labels.currentIndex() == 1:
            return self.table_orientation
        elif self.tab_sheet_labels.currentIndex() == 2:
            return self.table_interaction
        else:
            assert False

    def get_current_default_label(self):
        if self.tab_sheet_labels.currentIndex() == 0:
            return self.LBL_WALK
        elif self.tab_sheet_labels.currentIndex() == 1:
            return self.LBL_STRAIGHT
        elif self.tab_sheet_labels.currentIndex() == 2:
            return self.LBL_NO
        else:
            assert False

    def get_table_to_lbl_column(self, lbl_column):
        if lbl_column == self.LBL_COLUMN_MODE:
            return self.table_mode
        elif lbl_column == self.LBL_COLUMN_ORIENTATION:
            return self.table_orientation
        elif lbl_column == self.LBL_COLUMN_INTERACTION:
            return self.table_interaction

    def get_default_label_to_lbl_column(self, lbl_column):
        if lbl_column == self.LBL_COLUMN_MODE:
            return self.LBL_WALK
        elif lbl_column == self.LBL_COLUMN_ORIENTATION:
            return self.LBL_STRAIGHT
        elif lbl_column == self.LBL_COLUMN_INTERACTION:
            return self.LBL_NO
        else:
            assert False

    def table_double_click(self, mi):
        current_table = self.get_current_table()
        row_idx = mi.row()
        self.playback_bar.select_value(int(current_table.item(row_idx, 0).text()))

    def button_remove_click(self):
        current_table = self.get_current_table()
        indexes = current_table.selectionModel().selectedRows()
        for index in sorted(indexes):
            current_table.removeRow(index.row())
        self.label_text.setText('label %s' % (self.get_current_label()))

    def button_add_click(self, label=None):
        current_table = self.get_current_table()
        row_count = current_table.rowCount()  # necessary even when there are no rows in the table
        current_frame = self.playback_bar.selected_value()
        target_row_index = 0
        for i in range(row_count):
            if current_table.item(i, 0) is None or current_table.item(i, 0).text() != '' and int(
                    current_table.item(i, 0).text()) < current_frame:
                target_row_index = i + 1
        current_table.insertRow(target_row_index)
        current_table.setItem(target_row_index, 0, QTableWidgetItem(str(current_frame)))
        current_table.setItem(target_row_index, 1, QTableWidgetItem(label if label else self.combobox_label.currentText()))
        self.label_text.setText('label %s' % (self.get_current_label()))

    def button_export_click(self):
        save_filepath, _ = QFileDialog.getSaveFileName(self, 'Save labeled dataframe', self.subject_path,
                                                       'csv files (*.csv)', options=QFileDialog.DontUseNativeDialog)
        if save_filepath != '':
            for column in [self.LBL_COLUMN_MODE, self.LBL_COLUMN_ORIENTATION, self.LBL_COLUMN_INTERACTION]:
                labels = []
                last_index = 0
                last_label = self.get_default_label_to_lbl_column(column)
                table = self.get_table_to_lbl_column(column)
                for i in range(table.rowCount()):
                    if table.item(i, 0) is None or table.item(i, 0).text() == '':
                        break
                    else:
                        labels += (int(table.item(i, 0).text()) - last_index) * [last_label]
                        last_index = int(table.item(i, 0).text())
                        last_label = table.item(i, 1).text()
                labels += (self.labels_data_frame.index[-1] + 1 - last_index) * [last_label]
                self.labels_data_frame[column] = labels
            self.labels_data_frame.to_csv(save_filepath, index=False)
            os.chmod(save_filepath, 0o660)
            print('saved labeled data frame %s' % save_filepath)

    def init_labels(self):
        for column in [self.LBL_COLUMN_MODE, self.LBL_COLUMN_ORIENTATION, self.LBL_COLUMN_INTERACTION]:
            table = self.get_table_to_lbl_column(column)
            if column not in self.labels_data_frame.columns:
                self.labels_data_frame[column] = self.get_default_label_to_lbl_column(column)
            labels = self.labels_data_frame[column]
            last_label = ''
            for idx, label in enumerate(labels):
                if label != last_label:
                    row_idx = table.rowCount()
                    table.insertRow(row_idx)
                    table.setItem(row_idx, 0, QTableWidgetItem(str(idx)))
                    table.setItem(row_idx, 1, QTableWidgetItem(label))
                    last_label = label

    def get_current_label(self):
        current_table = self.get_current_table()
        frame_idx = self.playback_bar.selected_value()
        last_label = self.get_current_default_label()
        last_index = 0
        for i in range(current_table.rowCount()):
            if current_table.item(i, 0) is None or current_table.item(i, 0).text() == '':
                break
            else:
                current_index = int(current_table.item(i, 0).text())
                if last_index <= frame_idx < current_index:
                    return last_label
                last_label = current_table.item(i, 1).text()
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
        bottom_layout.addWidget(self.tab_sheet_labels)

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
