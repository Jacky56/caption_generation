from typing import List, Dict

import numpy as np


# https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
# https://github.com/keras-team/keras/issues/1627
def generate_data(filenames: List[str], filename_image_vector: Dict[str, List[float]],
                  filename_text_vector: Dict[str, List[str]], text_input_name, image_input_name):
    i = 0

    dataset_size = len(filenames)
    while True:
        image_vector_batch = []
        text_vector_batch = []

        image_vector = filename_image_vector[filenames[i % dataset_size]]
        text_vectors = filename_text_vector[filenames[i % dataset_size]]

        # image_vector_batch.append(image_vector)
        # too lazy to random
        # text_vector_batch.append(text_vectors[(i+3) % len(text_vectors)])
        text_vector = text_vectors[(i+3) % len(text_vectors)]
        # text_vector = text_vectors[0]
        i += 1
        
        inputs = {
            text_input_name: np.expand_dims(text_vector[:-1], 0),
            image_input_name: np.expand_dims(image_vector, 0)
        }
        outputs = np.expand_dims(text_vector[1:], 0)

        yield inputs, outputs

