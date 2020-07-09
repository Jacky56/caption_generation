from customlayers import imageencoder, textdecoder, positionencoding, decoder, encoderdecoder
from tensorflow.keras.layers import Conv1D, Input
from tensorflow.keras.models import Model
from staticvariables import statics
import numpy as np
import unittest
from custommodels import imagecaption


class TestCustomModel(unittest.TestCase):

    def test_init_model(self):
        repeat_layers = 6
        embedding_dim = 256
        target_vocab_size = 60
        image_feature_size = 10
        model = imagecaption.ImageCaption(repeat_layers, embedding_dim,
                                          target_vocab_size, image_feature_size)

        # seq (batch_size, seq_length)
        sparse_sequence_vector = np.ones((1, 10))
        # image (batch_size, dim)
        image_vector = np.ones((1, image_feature_size))

        # to name inputs
        input_text_key = 'text_input'
        input_image_key = 'image_input'
        inputs_data = {
            input_text_key: sparse_sequence_vector,
            input_image_key: image_vector,
        }
        val = model().predict(inputs_data)

        print(val)
        print(model.input_map)
        print(model().summary())

