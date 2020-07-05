from typing import List, Dict

import numpy as np


# https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
# https://github.com/keras-team/keras/issues/1627
def generate_data(filenames: List[str], batch_size: int,
                  filename_image_vector: Dict[str, str],
                  filename_text_vector: Dict[str, List[str]]):
    i = 0
    training_size = len(filenames)
    while True:
        image_vector_batch = []
        text_vector_batch = []
        for b in range(batch_size):
            image_vector = filename_image_vector[filenames[i % training_size]]
            text_vectors = filename_text_vector[filenames[i % training_size]]
            i += 1

            image_vector_batch.append(image_vector)
            # too lazy to random
            text_vector_batch.append(text_vectors[(i+3) % len(text_vectors)])

        yield np.array(image_vector_batch), np.array(text_vector_batch)
        # yield [np.array(image_vector_batch), np.array(text_vector_batch)], np.array(text_vector_batch)
