import tensorflow as tf


def build_model():
    model = tf.keras.applications.vgg16.VGG16()
    model.layers.pop()
    model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-1].output)

    return model


if __name__ == '__main__':
    model = build_model()
    print(model.layers[0].input.shape[1:3])
    pass