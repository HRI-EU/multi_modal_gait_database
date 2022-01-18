# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
import unittest

from src.data_frame_extractor import DataFrameExtractor, IMU_PREFIX, EYE_TRACKER_PREFIX, INSOLES_PREFIX, ALL_SUBJECT_IDS
from src.common import COURSE_A, COURSE_B, COURSE_C


class TestExtractor(unittest.TestCase):
    SOURCE_PATH = "/home/vlosing/storage/phume/gait_data_set"

    def test_data_frame_creation(self):
        data_manager = DataFrameExtractor(TestExtractor.SOURCE_PATH, [COURSE_A, COURSE_B, COURSE_C], [IMU_PREFIX, EYE_TRACKER_PREFIX, INSOLES_PREFIX], ALL_SUBJECT_IDS[:1])
        data_manager.create_data_frame()


if __name__ == '__main__':
    import logging
    import unittest
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    unittest.main()
