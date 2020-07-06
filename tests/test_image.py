import pytest
from preprocessingimages.app import *
from staticvariables import statics


def test_load_pkl_file():
    directory = statics.DATA_PATH
    features = load_filename_feature("{}/filename_features_4096.pkl".format(directory))

    print(len(features))

    for k, v in features.items():
        print(k, v)
        break



