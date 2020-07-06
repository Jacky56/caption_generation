from tensorflow.keras.layers import Dense, Attention, LayerNormalization, Dropout, Layer

class Text_Decoder_Layer(Layer):
    def __init__(self, embedding_dim, dff=512, dropout=0.1):
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

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, decoder_output, encoder_output, training=True):

        attention1 = self.attention1([decoder_output, decoder_output])
        drop1 = self.dropout1(attention1, training=training)
        norm1 = self.layer_norm(drop1 + decoder_output)

        attention2 = self.attention2([norm1, encoder_output])
        drop2 = self.dropout2(attention2, training=training)
        norm2 = self.layer_norm(drop2 + norm1)

        ffn1 = self.dff(norm2)
        ffn2 = self.ffn(ffn1)

        out1 = self.layer_norm(ffn2 + norm2)

        return out1
        # return out2, self.attention.weights

