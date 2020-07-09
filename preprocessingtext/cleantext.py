from typing import List, Dict
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from staticvariables import statics
from logger import Logger
from pickle import dump, load


def build_tokenizer(corpus: List[List[str]], num_words=None, oov_token='<unk>') -> Tokenizer:

    # flat map from List[List[str]] to List[str]
    # removed <> in filters
    texts = [text for texts in corpus for text in texts]
    tokenizer = Tokenizer(
        num_words=num_words,
        oov_token=oov_token,
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
    )
    tokenizer.fit_on_texts(texts)
    # id 0 is not set, it is supposed to be set by user with token
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer


def store_tokenizer(file_path: str, tokenizer: Tokenizer) -> None:
    dump(tokenizer.to_json(), open(file_path, 'wb'))


def load_tokenizer(file_path: str) -> Tokenizer:
    return tokenizer_from_json(load(open(file_path, 'rb')))


@Logger('load_text', directory=statics.LOGGING_PATH)
def load_text(filename, directory=statics.DATA_PATH) -> str:
    file = open('{}/{}'.format(directory, filename), 'r')
    text = file.read()
    file.close()

    return text


# create a dict with key: filename val: raw_text
# return dict: str -> List[str]
def create_filename_text(raw_text: str, start_token='<start>', end_token='<end>') -> Dict[str, List[str]]:
    filename_text = {}

    for line in raw_text.split('\n'):
        if len(line.strip()) < 2:
            continue

        words = line.split()
        filename = words[0].split('.')[0]
        text = ' '.join([start_token] + words[1:] + [end_token])

        # text format will contain one [filename] to many [text]
        if filename not in filename_text:
            filename_text[filename] = []
        filename_text[filename].append(text)

    return filename_text


# nominal to sparse vector
# return dict: str -> List[str]
def vectorize_filename_text(filename_text: dict, tokenizer: Tokenizer) -> Dict[str, List[str]]:
    for filename in filename_text:
        filename_text[filename] = tokenizer.texts_to_sequences(filename_text[filename])
        # filename_text[filename] = pad_sequences(filename_text[filename], padding='post')

    return filename_text


def texterize_filename_sparse_vector(filename_sparse_vector, okenizer: Tokenizer):
    pass
