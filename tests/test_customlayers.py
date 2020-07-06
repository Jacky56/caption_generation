import pytest
from customlayers import imageencoder, textdecoder, positionencoding


import numpy as np

def test_custom_encoder():
    model = imageencoder.Image_Encoder(16)

    # input (batch_size, input_size)
    print(model(np.ones((2, 19))).shape)


def test_custom_decoder():
    embedding_dim = 16
    seq_length = 10
    encoder = imageencoder.Image_Encoder(embedding_dim)
    decoder = textdecoder.Text_Decoder_Layer(embedding_dim)

    # input (batch_size, input_size)
    encoder_out = encoder(np.ones((1, 128)))
    # input (batch_size, 1, embedding_dim)

    # seq (batch_size, seq_length, embedding_dim)
    seq = np.ones((1, seq_length, embedding_dim))

    decoder_out = decoder(seq, encoder_out)
    print(decoder_out)

    decoder_out = decoder(decoder_out, encoder_out)
    print(decoder_out)

def test_positional_encoding():
    embedding_dim = 16
    seq_length = 10
    maximum_position_encoding = 5000
    val = positionencoding.positional_encoding(maximum_position_encoding, embedding_dim)
    print(val[:, :seq_length, :])
