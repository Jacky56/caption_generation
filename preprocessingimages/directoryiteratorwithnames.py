from tensorflow.keras.preprocessing.image import DirectoryIterator
import numpy as np
from os.path import basename


# https://stackoverflow.com/questions/41715025/keras-flowfromdirectory-get-file-names-as-they-are-being-generated
class DirectoryIteratorWithNames(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # format to only file name
        self.filenames_np = np.array([basename(e).split('.')[0] for e in self.filenames])
        # self.class_mode = None

    # override method and return tuple with name
    def _get_batches_of_transformed_samples(self, index_array) -> tuple:
        return self.filenames_np[index_array], super()._get_batches_of_transformed_samples(index_array)

