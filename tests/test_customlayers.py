import pytest
from customlayers import imageencoder, textdecoder, positionencoding, decoder
from custommodels import encoderdecoder
from tensorflow.keras.layers import Conv1D
from staticvariables import statics
import numpy as np

def test_custom_encoder():
    embedding_dim = 16
    seq_length = 10

    model = imageencoder.Image_Encoder(embedding_dim)

    # input (batch_size, input_size)
    encoder_out = model(np.ones((1, 128)))
    # out (batch_size, 1, embedding_dim)

    # channels_first to expand dims from 1 to 10
    expand_dim = Conv1D(filters=seq_length, padding='same', kernel_size=1, data_format='channels_first')(encoder_out)
    # out (1 , seq_length, embedding_dim)

    print(expand_dim)


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


def test_decoder():
    embedding_dim = 16
    seq_length = 10
    dec = decoder.Decoder(6, embedding_dim, 6000)

    encoder = imageencoder.Image_Encoder(embedding_dim)

    # encoder_out (batch_size, 1, embedding_dim)
    encoder_out = encoder(np.ones((1, 128)))

    # seq (batch_size, seq_length)
    sparse_sequence_vector = np.ones((1, seq_length))

    val = dec(sparse_sequence_vector, encoder_out, False)

    print(val)


def test_encoder_decoder():
    embedding_dim = 256
    seq_length = 48
    vocab_size = 6000
    ende = encoderdecoder.Encoder_Decoder(4, embedding_dim, vocab_size)

    image_features = np.ones((1, 4096))
    sparse_sequence_vector = np.ones((1, seq_length))

    val = ende(sparse_sequence_vector, image_features, False)

    print(val)

    ende.summary()
    print(ende.trainable_variables)


def test_save_load_encoder_decoder():
    directory = statics.DATA_PATH
    embedding_dim = 256
    vocab_size = 6000

    ende = encoderdecoder.Encoder_Decoder(4, embedding_dim, vocab_size)
    ende.save_weights('{}/encoder_decoder_weights.test.h5'.format(directory), True)
    ende.load_weights('{}/encoder_decoder_weights.test.h5'.format(directory))



