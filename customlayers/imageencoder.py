
from tensorflow.keras.layers import Dense, Layer, Dropout
import tensorflow as tf

class Image_Encoder(Layer):
    def __init__(self, embedding_dim, dropout=0.1, activiation_type='relu'):
        super(Image_Encoder, self).__init__()
        self.dense = Dense(embedding_dim,
                           activation=activiation_type
                           )
        self.dropout1 = Dropout(dropout)

    def call(self, inputs, training=True):
        # (batch_size, context vector) -> (batch_size, embedding_dim)
        output = self.dense(inputs)
        # (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
        output = tf.expand_dims(output, 1)

        output = self.dropout1(output, training=training)

        return output

