import pytest
from customlayers import image_encoder
import numpy as np

def test_custom_encoder():
    model = image_encoder.Image_Encoder(16)

    # input (batch_size, input_size)
    print(model(np.ones((1, 19))))
