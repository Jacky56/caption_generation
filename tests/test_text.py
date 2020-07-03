import pytest
from pickle import load
from staticvariables import statics
from preprocessingimages.app import load_filename_feature
from preprocessingtext import cleantext


def test_vectorize_filename_text():
    raw_text = cleantext.load_text('Flickr8k.lemma.token.txt')

    filename_text = cleantext.create_filename_text(raw_text)
    tokenizer = cleantext.build_tokenizer(list(filename_text.values()))

    filename_vector = cleantext.vectorize_filename_text(filename_text, tokenizer)

    for k in filename_vector:
        print(k, filename_vector[k])
        break



