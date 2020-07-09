from preprocessingimages import imagegenerator, vgg
from tensorflow.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.models import Model
from pickle import dump, load
from logger import Logger
from staticvariables import statics


def create_filename_feature(model: Model, generator_it: DirectoryIterator):
    filename_feature = {}
    for batch in generator_it:
        filenames = batch[0]
        image_np = batch[1]
        features = model.predict(image_np, verbose=1)
        filename_feature.update(dict(zip(filenames, features)))
        # to prevent generator to default loop back to batch 0
        if len(filename_feature) == generator_it.n:
            break
        print(len(filename_feature))
    return filename_feature


@Logger("pickling_load", directory=statics.LOGGING_PATH)
def load_filename_feature(file_path: str) -> dict:
    return load(open(file_path, 'rb'))

@Logger("pickling", directory=statics.LOGGING_PATH)
def store_filename_feature(file_path: str, filename_feature: dict):
    dump(filename_feature, open(file_path, 'wb'))


if __name__ == '__main__':
    directory = statics.DATA_PATH

    model = vgg.build_model()
    target_size = model.layers[0].input.shape[1:3]
    generator_settings = imagegenerator.build_generator_settings()
    generator_iterator = imagegenerator.build_image_iterator(directory, generator_settings, target_size)

    # filename_feature = create_filename_feature(model, generator_iterator)
    # store_filename_feature("{}/filename_features_4096.pkl".format(directory), filename_feature)

    features = load_filename_feature("{}/filename_features_4096.pkl".format(directory))

    for k, v in features.items():
        print(k, v.shape)
        break




