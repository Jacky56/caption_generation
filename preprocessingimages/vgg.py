import tensorflow as tf

if __name__ == '__main__':
    pass


def build_model():
    model = tf.keras.applications.vgg16.VGG16()
    model.layers.pop()
    model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-1].output)

    return model
