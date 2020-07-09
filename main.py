from custommodels import imagecaption
from customlayers import iterator
from preprocessingtext import cleantext
from preprocessingimages import app
from staticvariables import statics
import numpy as np

if __name__ == '__main__':
    directory = statics.DATA_PATH

    batch_size = 32

    tokenizer = cleantext.load_tokenizer('{}/Flickr8k.lemma.token.pkl'.format(directory))
    raw_text = cleantext.load_text('Flickr8k.lemma.token.txt')
    filename_text = cleantext.create_filename_text(raw_text)

    # text data Dict[str, List[str]]
    filename_sparse_vector = cleantext.vectorize_filename_text(filename_text, tokenizer)

    # text data Dict[str, List[float]]
    filename_image_features = app.load_filename_feature("{}/filename_features_4096.pkl".format(directory))

    # list[str]
    filenames = list(filename_image_features.keys() & filename_sparse_vector.keys())

    repeat_layers = 6
    embedding_dim = 512
    target_vocab_size = len(tokenizer.word_counts)
    image_feature_size = 4096

    model = imagecaption.ImageCaption(repeat_layers, embedding_dim,
                                      target_vocab_size, image_feature_size)

    generator = iterator.generate_data(filenames, filename_image_features, filename_sparse_vector,
                                       model.text_input_name, model.image_input_name)



    data = next(generator)
    model().compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model().summary()

    model().fit(generator, steps_per_epoch=len(filenames)//32, epochs=10)
    print(filenames[0])


    print(data[0]['text_input'])
    ans = model().predict(data[0])

    img = data[0]['image_input']
    inputs = {
        'image_input': img,
        'text_input': np.expand_dims([3], 0)
    }
    for i in range(15):
        ans = model().predict(inputs)
        print(np.expand_dims(np.append(inputs['text_input'], ans.argmax(-1)[-1][-1]), 0))
        inputs['text_input'] = np.expand_dims(np.append(inputs['text_input'], ans.argmax(-1)[-1][-1]), 0)


    print(inputs['text_input'])










