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
import os

COURSE_A = "a"
COURSE_B = "b"
COURSE_C = "c"
EYE_TRACKER_FILE_NAME = 'eyetracker.csv'
INSOLES_FILE_NAME = 'insoles.csv'
IMU_FILE_NAME = 'xsens.csv'
LABELS_FILE_NAME = 'labels.csv'
VIDEO_FILE_NAME = 'video.mp4'

COURSE_TO_FOLDER_MAP = {COURSE_A: "courseA", COURSE_B: "courseB", COURSE_C: "courseC"}


def get_path_to_course_subject(source_path, course, subject_id):
    return os.path.join(os.path.join(source_path, COURSE_TO_FOLDER_MAP[course]), "id%s" % str(subject_id).zfill(2))
