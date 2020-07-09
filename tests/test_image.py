import pytest
import unittest
from preprocessingimages.app import *
from staticvariables import statics


class TestImage(unittest.TestCase):

    def test_load_pkl_file(self):
        directory = statics.DATA_PATH
        features = load_filename_feature("{}/filename_features_4096.pkl".format(directory))

        print(len(features))

        for k, v in features.items():
            print(k, v.shape)
            break


if __name__ == '__main__':
    unittest.main()


