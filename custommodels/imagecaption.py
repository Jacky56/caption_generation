from customlayers.encoderdecoder import EncoderDecoder
from customlayers.decoder import Decoder
from tensorflow.keras.layers import Dense, Layer, Input
from tensorflow.keras import Model


class ImageCaption:
    def __init__(self, repeat_layers, embedding_dim, target_vocab_size, image_feature_size,
                 dff=512, maximum_position_encoding=5000, dropout=0.1,
                 text_input_name='text_input',
                 image_input_name='image_input'):

        self.text_input_name = text_input_name
        self.image_input_name = image_input_name

        self.encoder_decoder = EncoderDecoder(repeat_layers, embedding_dim,
                                             target_vocab_size, dff,
                                             maximum_position_encoding,
                                             dropout)

        self.text_input = Input(shape=(None,), name=self.text_input_name)
        self.image_input = Input(shape=(image_feature_size,), name=self.image_input_name)

        self.caption_output = self.encoder_decoder(self.text_input, self.image_input)

        self.input_map = {
            self.text_input_name: self.text_input,
            self.image_input_name: self.image_input
        }

        self.model = Model(inputs=self.input_map, outputs=self.caption_output)

    def __call__(self) -> Model:
        return self.model


