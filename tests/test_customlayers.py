from customlayers import imageencoder, textdecoder, positionencoding, decoder, encoderdecoder
from tensorflow.keras.layers import Conv1D, Input
from tensorflow.keras.models import Model
from staticvariables import statics
import numpy as np
import unittest
from preprocessingtext import cleantext
from preprocessingimages import app
from customlayers import iterator

class TestCustomLayer(unittest.TestCase):

    def test_custom_encoder(self):
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

    def test_custom_decoder(self):
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

    def test_positional_encoding(self):
        embedding_dim = 16
        seq_length = 10
        maximum_position_encoding = 5000
        val = positionencoding.positional_encoding(maximum_position_encoding, embedding_dim)
        print(val[:, :seq_length, :])

    def test_decoder(self):
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

    def test_decoder_with_input(self):
        embedding_dim = 16
        seq_length = 10
        dec = decoder.Decoder(6, embedding_dim, 6000)

        encoder = imageencoder.Image_Encoder(embedding_dim)

        # seq (batch_size, seq_length)
        sparse_sequence_vector = np.ones((1, seq_length))
        # image (batch_size, dim)
        image_vector = np.ones((1, 128))

        # to name inputs 
        input_text_key = 'ins_text'
        input_image_key = 'ins_image'
        # none allows variable input
        ins_text = Input(shape=(None,), name=input_text_key)
        ins_image = Input(shape=(128,), name=input_image_key)

        inputs = {
            input_text_key: ins_text,
            input_image_key: ins_image,
        }

        encoder_out = encoder(ins_image)
        outs = dec(ins_text, encoder_out, False)
        model = Model(inputs=inputs, outputs=[outs])

        inputs_data = {
            input_text_key: sparse_sequence_vector,
            input_image_key: image_vector,
        }

        ans = model.predict(inputs_data)

        print(ans)
        print(ans.shape)

    def test_encoder_decoder(self):
        embedding_dim = 256
        seq_length = 48
        vocab_size = 6000
        ende = encoderdecoder.EncoderDecoder(4, embedding_dim, vocab_size)

        image_features = np.ones((1, 4096))
        sparse_sequence_vector = np.ones((1, seq_length))

        val = ende(sparse_sequence_vector, image_features, False)

        print(val)

        # ende.summary() only Model can do this


        print(ende.trainable_variables)

    # only models can do this
    # def test_save_load_encoder_decoder(self):
    #     directory = statics.DATA_PATH
    #     embedding_dim = 256
    #     vocab_size = 6000
    #
    #     ende = encoderdecoder.EncoderDecoder(4, embedding_dim, vocab_size)
    #     ende.save_weights('{}/encoder_decoder_weights.test.h5'.format(directory), True)
    #     ende.load_weights('{}/encoder_decoder_weights.test.h5'.format(directory))



    def test_iterator(self):
        directory = statics.DATA_PATH

        batch_size = 32
        seq_length = 36

        tokenizer = cleantext.load_tokenizer('{}/Flickr8k.lemma.token.pkl'.format(directory))
        raw_text = cleantext.load_text('Flickr8k.lemma.token.txt')
        filename_text = cleantext.create_filename_text(raw_text)
        # text data Dict[str, List[str]]
        filename_sparse_vector = cleantext.vectorize_filename_text(filename_text, tokenizer)

        # text data Dict[str, List[float]]
        filename_image_features = app.load_filename_feature("{}/filename_features_4096.pkl".format(directory))

        # list[str]
        filenames = list(filename_image_features.keys() & filename_sparse_vector.keys())[:100]

        generator = iterator.generate_data(filenames, filename_image_features, filename_sparse_vector, 'text', 'image')

        data = [next(generator) for i in range(10)]





if __name__ == '__main__':
    unittest.main()
