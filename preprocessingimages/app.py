import tensorflow as tf
from preprocessingimages import generator, vgg
from tensorflow.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.models import Model
from pickle import dump
from preprocessingimages.logger import Logger
from staticvariables import statics


def create_filename_feature(model: Model, generator_it: DirectoryIterator):
    filename_feature = {}

    for batch in generator_it:
        filenames = batch[0]
        image_np = batch[1]
        features = model.predict(image_np)
        filename_feature.update(dict(zip(filenames, features)))

    return filename_feature


@Logger("pickling", directory=statics.LOGGING_PATH)
def store_filename_feature(file_path: str, filename_feature: dict):
    dump(filename_feature, open(file_path, 'wb'))


if __name__ == '__main__':
    directory = statics.DATA_PATH

    model = vgg.build_model()
    target_size = model.layers[0].input.shape[1:3]
    generator_settings = generator.build_generator_settings()
    generator_iterator = generator.build_image_iterator(directory, generator_settings, target_size)

    filename_feature = create_filename_feature(model, generator_iterator)

    store_filename_feature("{}/filename_features_4096.pkl".format(directory), filename_feature)





