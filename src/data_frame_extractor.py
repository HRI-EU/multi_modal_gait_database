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

import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import EYE_TRACKER_FILE_NAME, INSOLES_FILE_NAME, IMU_FILE_NAME, LABELS_FILE_NAME, \
    get_path_to_course_subject, COURSE_A, COURSE_B, COURSE_C

POSITION_SEGMENTS = ['Head', 'L3', 'L5', 'LeftFoot',
                     'LeftForeArm', 'LeftHand', 'LeftLowerLeg', 'LeftShoulder',
                     'LeftToe', 'LeftUpperArm', 'LeftUpperLeg', 'Neck', 'Pelvis', 'RightFoot',
                     'RightForeArm', 'RightHand', 'RightLowerLeg', 'RightShoulder', 'RightToe',
                     'RightUpperArm', 'RightUpperLeg', 'T12', 'T8']

POSITION_PREFIX = 'position'

IMU_PREFIX = "imu"
INSOLES_PREFIX = "insoles"
EYE_TRACKER_PREFIX = "eyetracker"
LABELS_PREFIX = "labels"

SEQUENCE_ID_COLUMN = 'sequence_id'

ALL_SUBJECT_IDS = [1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25]

FILE_NAME_TO_COLUMN_PREFIX_MAP = {EYE_TRACKER_FILE_NAME: EYE_TRACKER_PREFIX, INSOLES_FILE_NAME: INSOLES_PREFIX,
                                  IMU_FILE_NAME: IMU_PREFIX, LABELS_FILE_NAME: LABELS_PREFIX}
LABEL_COLUMN = "%s_%s" % (LABELS_PREFIX, "walk_mode")
DUPLICATE_COLUMNS = ['time', 'participant_id', 'task']


def get_position_column_name_x(segment):
    return '%s_%s_%s_%s' % (IMU_PREFIX, POSITION_PREFIX, segment, 'x')


def get_position_column_name_y(segment):
    return '%s_%s_%s_%s' % (IMU_PREFIX, POSITION_PREFIX, segment, 'y')


class DataFrameExtractor:
    def __init__(self, source_path, courses, modalities, subject_ids, float_32=True, lower_columns=True,
                 center_positions=True):
        assert len(courses) > 0, "at least one course required"
        assert len(modalities) > 0, "at least one modality required"
        assert len(subject_ids) > 0, "at least one subject id required"
        self.source_path = source_path
        self.courses = courses
        self.modalities = modalities
        self.subject_ids = subject_ids
        self.float_32 = float_32
        self.lower_columns = lower_columns
        self.center_positions = center_positions

    @staticmethod
    def centre_positions(data_frame):
        result_df = data_frame.copy(True)

        column_right_foot_x = get_position_column_name_x('RightFoot')
        column_right_foot_y = get_position_column_name_y('RightFoot')
        column_left_foot_x = get_position_column_name_x('LeftFoot')
        column_left_foot_y = get_position_column_name_y('LeftFoot')

        center_offset = (data_frame[[column_right_foot_x, column_right_foot_y]].to_numpy() + data_frame[
            [column_left_foot_x, column_left_foot_y]].to_numpy()) * 0.5

        column_com_x = '%s_centerOfMass_x' % IMU_PREFIX
        column_com_y = '%s_centerOfMass_y' % IMU_PREFIX
        result_df[column_com_x] -= center_offset[:, 0]
        result_df[column_com_y] -= center_offset[:, 1]

        for segment in POSITION_SEGMENTS:
            column_x = get_position_column_name_x(segment)
            column_y = get_position_column_name_y(segment)
            result_df[column_x] -= center_offset[:, 0]
            result_df[column_y] -= center_offset[:, 1]
        return result_df

    @staticmethod
    def get_sequence_ids(labels, id_offset):
        last_class = -1
        last_change_idx = 0
        ids = []
        id = id_offset + 1

        for idx in range(len(labels)):
            if idx == 0:
                last_class = labels[idx]
            elif last_class != labels[idx]:
                ids += [id] * (idx - last_change_idx)
                id += 1
                last_change_idx = idx
                last_class = labels[idx]
        ids += [id] * (len(labels) - last_change_idx)
        return ids

    def get_subject_path(self, course_path, subject_id):
        return os.path.join(course_path, "id%s" % str(subject_id).zfill(2))

    def file_name_to_modality(self, file_name):
        return FILE_NAME_TO_COLUMN_PREFIX_MAP[file_name] if file_name in FILE_NAME_TO_COLUMN_PREFIX_MAP else None

    def create_data_frame(self):
        """Creates data frame using all subdirectories of "sourcePath". Extracts all motions of the data.
        """
        logging.info('create walking data frame')
        if not os.path.exists(self.source_path):
            raise Exception("source path for db does not exist %s" % self.source_path)
        sequence_id_offset = 0
        all_data_frames = []

        course_bar = tqdm(self.courses)
        for course in course_bar:
            course_bar.set_description("processing course %s" % course)
            subject_bar = tqdm(self.subject_ids)
            for subject_id in subject_bar:
                print(subject_id)
                subject_bar.set_description("processing subject %d" % subject_id)
                subject_data_frames = []
                subject_path = get_path_to_course_subject(self.source_path, course, subject_id)
                for root, dir_list, file_list in os.walk(subject_path):
                    for filename in np.sort(file_list):
                        if filename == LABELS_FILE_NAME or self.file_name_to_modality(filename) in self.modalities:
                            data_frame = pd.read_csv(os.path.join(root, filename))
                            column_prefix = FILE_NAME_TO_COLUMN_PREFIX_MAP[filename]
                            column_mapping = {column: "%s_%s" % (column_prefix, column) for column in
                                              data_frame.columns if
                                              column not in DUPLICATE_COLUMNS}
                            if filename == LABELS_FILE_NAME:
                                for column in DUPLICATE_COLUMNS:
                                    column_mapping[column] = column
                            data_frame.rename(columns=column_mapping, inplace=True)
                            subject_data_frames.append(data_frame)
                    if len(subject_data_frames) > 0:
                        subject_data_frame = pd.concat(subject_data_frames, axis=1)
                        subject_data_frame[SEQUENCE_ID_COLUMN] = self.get_sequence_ids(
                            subject_data_frame[LABEL_COLUMN].tolist(), sequence_id_offset)
                        if self.float_32:
                            columns = subject_data_frame.columns[
                                (subject_data_frame.dtypes.values == np.dtype('float64'))]
                            subject_data_frame[columns] = subject_data_frame[columns].astype(np.float32)
                        all_data_frames.append(subject_data_frame)
                        sequence_id_offset = subject_data_frame.iloc[-1][SEQUENCE_ID_COLUMN]

        data_frame = pd.concat(all_data_frames, axis=0, ignore_index=True)
        if IMU_PREFIX in self.modalities and self.center_positions:
            data_frame = self.centre_positions(data_frame)
        if self.lower_columns:
            data_frame.columns = data_frame.columns.str.lower()
        # magnetic_field_columns = [col for col in data_frame.columns if 'sensormagneticfield' in col]
        # data_frame.drop(columns=magnetic_field_columns, inplace=True)
        logging.info('dataframe generated')
        return data_frame


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    argparser = argparse.ArgumentParser(description='pandas data frame creation script.')
    argparser.add_argument('--source_path', '-p', action='store', required=True,
                           dest='source_path', type=str,
                           help='path to dataset folder')
    argparser.add_argument('--dest_path', '-d', action='store', required=True,
                           dest='dest_path', type=str,
                           help='path for created data frame')
    argparser.add_argument('--course', '-c', action='store', required=False, default="",
                           dest='courses', type=str,
                           help='include specific courses(%s, %s, %s) given as comma-separeted list(default is to include all courses)' % (
                               COURSE_A, COURSE_B, COURSE_C))
    argparser.add_argument('--modality', '-m', action='store', required=False, default="",
                           dest='modalities', type=str,
                           help='include specific modalities(%s, %s, %s) given as comma-separeted list(default is to include all modalities)' % (
                               IMU_PREFIX, INSOLES_PREFIX, EYE_TRACKER_PREFIX))
    argparser.add_argument('--subject', '-s', action='store', required=False, default="",
                           dest='subject_ids', type=str,
                           help='include specific subject ids given as comma-seperated list (default is to include all subjects)')
    argparser.add_argument('--float32', '-f', action='store_true', dest='float_32', help='skip synchronization')
    argparser.add_argument('--lower_columns', '-l', action='store_true', dest='lower_columns',
                           help='lower all column names')
    argparser.add_argument('--center_positions', '-e', action='store_true', dest='center_positions',
                           help='center positions into midpoint between left and right foot')

    args = argparser.parse_args()

    if args.courses == "":
        courses = [COURSE_A, COURSE_B, COURSE_C]
    else:
        courses = args.courses.replace(" ", "").split(",")

    if args.modalities == "":
        modalities = [IMU_PREFIX, INSOLES_PREFIX, EYE_TRACKER_PREFIX]
    else:
        modalities = args.modalities.replace(" ", "").split(",")

    if args.subject_ids == "":
        subject_ids = ALL_SUBJECT_IDS
    else:
        subject_ids = args.subject_ids.replace(" ", "").split(",")
        subject_ids = [int(elem) for elem in subject_ids]

    data_manager = DataFrameExtractor(args.source_path, courses, modalities, subject_ids, args.float_32,
                                      args.lower_columns, args.center_positions)
    data_frame = data_manager.create_data_frame()
    data_frame.to_pickle(args.dest_path)
