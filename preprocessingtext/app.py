from preprocessingtext import cleantext
from staticvariables import statics

# creates new token file pkl
if __name__ == '__main__':
    directory = statics.DATA_PATH

    raw_text = cleantext.load_text('Flickr8k.lemma.token.txt')
    filename_text = cleantext.create_filename_text(raw_text)
    tokenizer = cleantext.build_tokenizer(list(filename_text.values()))
    cleantext.store_tokenizer('{}/Flickr8k.lemma.token.pkl'.format(directory), tokenizer)
    tokenizer1 = cleantext.load_tokenizer('{}/Flickr8k.lemma.token.pkl'.format(directory))



    # print(tokenizer.word_counts == tokenizer1.word_counts)
    # print(tokenizer.word_index['<start>'] == tokenizer1.word_index['<start>'])
    # print(tokenizer.word_index['<end>'] == tokenizer1.word_index['<end>'])
    # print(tokenizer.word_index['<pad>'] == tokenizer1.word_index['<pad>'])



