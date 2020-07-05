from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class Image_Encoder(Model):
    def __init__(self, embedding_dim, activiation_type='relu'):
        super(Image_Encoder, self).__init__()
        self.dense = Dense(embedding_dim,
                           activation=activiation_type
                           )

    def call(self, inputs):
        output = self.dense(inputs)
        return output