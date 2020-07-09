from customlayers.textdecoder import Text_Decoder_Layer
from customlayers.positionencoding import positional_encoding
from tensorflow.keras.layers import Layer, Embedding, Dropout
import tensorflow as tf

class Decoder(Layer):
    def __init__(self, repeat_layers, embedding_dim, target_vocab_size,
                 dff=512, maximum_position_encoding=5000, dropout=0.1):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.repeat_layers = repeat_layers

        self.embedding = Embedding(target_vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, embedding_dim)

        self.dec_layers = [Text_Decoder_Layer(embedding_dim, dff, dropout) for _ in range(repeat_layers)]

        self.dropout = Dropout(dropout)

    def call(self, sparse_sequence_vector, encoder_output, training=True):

        # seq_len is varaible len
        seq_len = tf.shape(sparse_sequence_vector)[1]

        embedding_output = self.embedding(sparse_sequence_vector)
        embedding_output *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        embedding_output += self.pos_encoding[:, :seq_len, :]

        decoder_output = self.dropout(embedding_output, training=training)

        for dec_layer in self.dec_layers:
            decoder_output = dec_layer(decoder_output, encoder_output, training)

        # (batch_size, sequence_length, embedding_dim)
        return decoder_output

