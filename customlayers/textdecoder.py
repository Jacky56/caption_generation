from tensorflow.keras.layers import Dense, Attention, LayerNormalization, Dropout, Layer
import tensorflow as tf


class Text_Decoder_Layer(Layer):
    def __init__(self, embedding_dim, dff=512, drop_out=0.1):
        super(Text_Decoder_Layer, self).__init__()

        self.attention1 = Attention(use_scale=True,
                                   causal=True
                                   )

        self.attention2 = Attention(use_scale=True,
                                   causal=True
                                   )

        self.dff = Dense(dff, activation='relu')
        self.ffn = Dense(embedding_dim)

        self.layer_norm = LayerNormalization()

        self.dropout1 = Dropout(drop_out)
        self.dropout2 = Dropout(drop_out)

    def call(self, x, enc_output):

        attention1 = self.attention1([x, x])
        drop1 = self.dropout1(attention1)
        norm1 = self.layer_norm(drop1 + x)

        attention2 = self.attention2([norm1, enc_output])
        drop2 = self.dropout2(attention2)
        norm2 = self.layer_norm(drop2 + norm1)

        ffn1 = self.dff(norm2)
        ffn2 = self.ffn(ffn1)

        out1 = self.layer_norm(ffn2 + norm2)

        return out1
        # return out2, self.attention.weights

