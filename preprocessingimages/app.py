import tensorflow as tf
from . import generator, vgg


if __name__ == '__main__':
    directory = " "
    model = vgg.build_model()
    target_size = model.layers[0].input.shape[1:3]
    generator_settings = generator.build_generator_settings()
    generator_iterator = generator.build_image_iterator(directory, generator_settings, target_size)



