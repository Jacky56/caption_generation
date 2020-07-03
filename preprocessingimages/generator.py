from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessingimages.directoryiteratorwithnames import DirectoryIteratorWithNames
from preprocessingimages.logger import Logger
from staticvariables import statics

def build_generator_settings() -> ImageDataGenerator:
    settings = ImageDataGenerator(

    )

    return settings



@Logger("generator", directory=statics.LOGGING_PATH)
def build_image_iterator(directory, generator_settings, target_size) -> DirectoryIteratorWithNames:
    iterator = DirectoryIteratorWithNames(
        directory,
        generator_settings,
        target_size,
        class_mode=None
    )
    return iterator


if __name__ == '__main__':
    gen_setting = build_generator_settings()
    iterator = build_image_iterator('E:/Data/Flicker8k', gen_setting, (224, 224))
