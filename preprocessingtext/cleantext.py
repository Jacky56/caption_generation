from typing import List
from tensorflow.keras.preprocessing.text import Tokenizer
from staticvariables import statics
from logger import Logger


def build_tokenizer(corpus: List[List[str]], num_words=5000, oov_token='<unk>'):

    # flat map from List[List[str]] to List[str]
    texts = [text for texts in corpus for text in texts]
    tokenizer = Tokenizer(
        num_words=num_words,
        oov_token=oov_token
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


@Logger('load_text', directory=statics.LOGGING_PATH)
def load_text(filename, directory=statics.DATA_PATH):
    file = open('{}/{}'.format(directory, filename), 'r')
    text = file.read()
    file.close()

    return text


# create a dict with key: filename val: raw_text
def create_filename_text(raw_text: str) -> dict:
    filename_text = {}

    for line in raw_text.split('\n'):
        if len(line.strip()) < 2:
            continue

        words = line.split()
        filename = words[0].split('.')[0]
        text = ' '.join(words[1:])
        if filename not in filename_text:
            filename_text[filename] = []
        filename_text[filename].append(text)

    return filename_text


def vectorize_filename_text(filename_text: dict, tokenizer: Tokenizer) -> dict:

    for filename in filename_text:
        filename_text[filename] = tokenizer.texts_to_sequences(filename_text[filename])

    return filename_text


