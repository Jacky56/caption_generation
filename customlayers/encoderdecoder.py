from customlayers.imageencoder import Image_Encoder
from customlayers.decoder import Decoder
from tensorflow.keras.layers import Dense, Layer


class EncoderDecoder(Layer):
    def __init__(self, repeat_layers, embedding_dim, target_vocab_size,
                 dff=512, maximum_position_encoding=5000, dropout=0.1):
        super(EncoderDecoder, self).__init__()

        self.encoder = Image_Encoder(embedding_dim, dropout)
        self.decoder = Decoder(repeat_layers, embedding_dim, target_vocab_size,
                               dff, maximum_position_encoding, dropout)

        self.ffn = Dense(target_vocab_size)


    def call(self, sparse_sequence_vector, image_features, training=True):

        # (batch_size, 1, embedding_dim)
        encoder_output = self.encoder(image_features)

        # (batch_size, sequence_length, embedding_dim)
        decoder_output = self.decoder(sparse_sequence_vector, encoder_output)

        # (batch_size, sequence_length, target_vocab_size)
        final_output = self.ffn(decoder_output)

        return final_output

