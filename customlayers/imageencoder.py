from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf

class Image_Encoder(Model):
    def __init__(self, embedding_dim, activiation_type='relu'):
        super(Image_Encoder, self).__init__()
        self.dense = Dense(embedding_dim,
                           activation=activiation_type
                           )

    def call(self, inputs):
        # (batch_size, context vector) -> (batch_size, embedding_dim)
        output = self.dense(inputs)
        # (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
        output = tf.expand_dims(output, 1)

        return output

