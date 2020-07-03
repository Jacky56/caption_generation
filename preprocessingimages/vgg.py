import tensorflow as tf
from tensorflow.keras.models import Model

def build_model() -> Model:
    model = tf.keras.applications.vgg16.VGG16()
    model.layers.pop()

    # remove last output layer, retaining context vector(4096)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    return model


if __name__ == '__main__':
    model = build_model()
    print(model.layers[0].input.shape[1:3])

    print(model.layers[-1].output.shape)

    pass

