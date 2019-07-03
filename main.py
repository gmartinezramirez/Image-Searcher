import os

import numpy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPRegressor

import pandas as pd

FILEPATH_TRAIN_IMAGES_NAME = "data/train/train_images_names.txt"
FILEPATH_TRAIN_IMAGES_VECTORS = "data/train/train_images_vectors.bin"
FILEPATH_TRAIN_CAPTION = "data/train/train_captions.txt"
FILEPATH_TEST_IMAGES_NAME = "data/test/test_A_images_names.txt"
FILEPATH_TEST_IMAGES_VECTORS = "data/test/test_A_images_vectors.bin"
FILEPATH_TEST_CAPTION = "data/test/test_A_captions.txt"
TRAIN_NUM_VECTORS = 20000
TRAIN_VECTOR_DIMENSIONS = 2048
TEST_NUM_VECTORS = 1000
TEST_VECTOR_DIMENSIONS = 2048


def load_file(file_names, file_vectors, num_vectors, vector_dimensions):
    assert os.path.isfile(file_names), "file doesnt exists " + file_names
    assert os.path.isfile(file_vectors), "file doesnt exists " + file_vectors
    print("Reading file of names: " + file_names)
    names = [line.strip() for line in open(file_names)]
    assert num_vectors == len(names), "len no compatible " + len(names)
    print("Reading file of vectors: " + file_vectors)
    mat = numpy.fromfile(file_vectors, dtype=numpy.float32)
    vectors = numpy.reshape(mat, (num_vectors, vector_dimensions))
    print(str(num_vectors) + " vector of size " + str(vector_dimensions))
    return names, vectors


def load_train_vectors():
    print("Load train vectors")
    return load_file(FILEPATH_TRAIN_IMAGES_NAME, FILEPATH_TRAIN_IMAGES_VECTORS, TRAIN_NUM_VECTORS,
                     TRAIN_VECTOR_DIMENSIONS)


def load_test_vectors():
    print("Load test vectors")
    return load_file(FILEPATH_TEST_IMAGES_NAME, FILEPATH_TEST_IMAGES_VECTORS, TEST_NUM_VECTORS, TEST_VECTOR_DIMENSIONS)


def load_captions(file_captions):
    assert os.path.isfile(file_captions), "file doesnt exists " + file_captions
    return [line.strip().split("\t") for line in open(file_captions, encoding='utf-8')]


def get_text_descriptor_by_tf_idf(texts):
    tf_idf_vect = TfidfVectorizer(lowercase=False, ngram_range=(1, 3), max_df=0.9, min_df=0.002, norm=None)
    tf_idf_vect.fit(texts)
    X_train = tf_idf_vect.transform(texts)
    print("Training: {}".format(X_train))
    print(X_train.shape)
    return X_train


def get_images_names_with_captions_grouped(dataframe):
    df = pd.DataFrame(dataframe, columns=['image_name', 'caption'])
    df_grouped_by_caption = df.groupby('image_name')['caption'].apply(
        ' '.join).reset_index()
    df_grouped_by_caption_with_image_name_as_index = df_grouped_by_caption.set_index('image_name')
    df_final = df_grouped_by_caption_with_image_name_as_index.reindex(train_names)
    return df_final


def get_all_captions_of_all_images_names(train_captions):
    captions_train = list()
    for filename, caption in train_captions:
        captions_train.append(caption)
    return captions_train


def classify_with_mlp_regression(x, y, test_vectors):
    neuronal_network = MLPRegressor(hidden_layer_sizes=(3), 
                  activation='tanh', solver='lbfgs')
    model_trained = neuronal_network.fit(x, y)
    print ("Train sucessful")
    neuronal_network.predict(test_vectors)
    print ("Predict sucessful")
    return model_trained

if __name__ == '__main__':
    (train_names, train_vectors) = load_train_vectors()
    (test_names, test_vectors) = load_test_vectors()

    train_captions = load_captions(FILEPATH_TRAIN_CAPTION)

    captions_train = get_images_names_with_captions_grouped(train_captions)
    captions_test = get_images_names_with_captions_grouped(train_captions)

    X_train = get_text_descriptor_by_tf_idf(captions_train['caption'])

    model_mlp_regression_trained = classify_with_mlp_regression(X_train, train_vectors, test_vectors)