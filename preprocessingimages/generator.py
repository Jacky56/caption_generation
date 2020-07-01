import tensorflow as tf
from preprocessingimages.logger import Logger


def build_generator_settings():
    settings = tf.keras.preprocessing.image.ImageDataGenerator(

    )

    return settings


@Logger("generator")
def build_image_iterator(directory, generator_settings, target_size):
    iterator = tf.keras.preprocessing.image.DirectoryIterator(
        directory,
        generator_settings,
        target_size,
    )

    return iterator


if __name__ == '__main__':
    gen_setting = build_generator_settings()
    iterator = build_image_iterator('.', gen_setting, (224, 224))
