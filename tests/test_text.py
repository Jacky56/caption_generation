import pytest
from staticvariables import statics
from preprocessingtext import cleantext


def test_vectorize_filename_text():
    raw_text = cleantext.load_text('Flickr8k.lemma.token.txt')
    filename_text = cleantext.create_filename_text(raw_text)
    tokenizer = cleantext.build_tokenizer(list(filename_text.values()))

    filename_vector = cleantext.vectorize_filename_text(filename_text, tokenizer, 64)

    print(tokenizer.index_word[144])
    print(tokenizer.index_word[669])
    print(len(tokenizer.word_counts))
    print(tokenizer.index_word[3])
    for k in filename_vector:
        print(k, filename_vector[k])
        break


def test_filename_text():
    raw_text = cleantext.load_text('Flickr8k.lemma.token.txt')
    filename_text = cleantext.create_filename_text(raw_text)

    for k in filename_text:
        print(k, filename_text[k])
        break



def test_load_tokenizer():
    directory = statics.DATA_PATH
    tokenizer_filename = 'Flickr8k.lemma.token.pkl.test'

    raw_text = cleantext.load_text('Flickr8k.lemma.token.txt')
    filename_text = cleantext.create_filename_text(raw_text)
    tokenizer = cleantext.build_tokenizer(list(filename_text.values()))
    cleantext.store_tokenizer('{}/{}'.format(directory, tokenizer_filename), tokenizer)
    tokenizer1 = cleantext.load_tokenizer('{}/{}'.format(directory, tokenizer_filename))

    print(tokenizer.word_counts == tokenizer1.word_counts)
    print(tokenizer.word_index['<start>'] == tokenizer1.word_index['<start>'])
    print(tokenizer.word_index['<end>'] == tokenizer1.word_index['<end>'])
    print(tokenizer.word_index['<pad>'] == tokenizer1.word_index['<pad>'])




