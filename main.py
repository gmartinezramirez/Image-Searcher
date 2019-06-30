import os

import numpy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model.logistic import LogisticRegression

import pandas as pd
import logging

FILEPATH_TRAIN_IMAGES_NAME = "data/train/train_images_names.txt"
FILEPATH_TRAIN_IMAGES_VECTORS = "data/train/train_images_vectors.bin"
FILEPATH_TRAIN_CAPTION = "data/train/train_captions.txt"
FILEPATH_TEST_IMAGES_NAME = "data/test/test_A_images_names.txt"
FILEPATH_TEST_IMAGES_VECTORS = "data/test/test_A_images_vectors.bin"
FILEPATH_TEST_CAPTION = "data/test/test_A_captions.txt"
TRAIN_NUM_VECTORS = 20000
TEST_NUM_VECTORS = 1000
TRAIN_VECTOR_DIMENSIONS = 2048
TEST_VECTOR_DIMENSIONS = 2048


def load_file(file_names, file_vectors, num_vectors, vector_dimensions):
    assert os.path.isfile(file_names), "file doesnt exists " + file_names
    assert os.path.isfile(file_vectors), "file doesnt exists " + file_vectors
    logging.info("Reading file of names: " + file_names)
    names = [line.strip() for line in open(file_names)]
    assert num_vectors == len(names), "len no compatible " + len(names)
    logging.info("Reading file of vectors: " + file_names)
    mat = numpy.fromfile(file_vectors, dtype=numpy.float32)
    vectors = numpy.reshape(mat, (num_vectors, vector_dimensions))
    print(str(num_vectors) + " vector of size " + str(vector_dimensions))
    return names, vectors


def load_train_vectors():
    return load_file(FILEPATH_TRAIN_IMAGES_NAME, FILEPATH_TRAIN_IMAGES_VECTORS, TRAIN_NUM_VECTORS,
                     TRAIN_VECTOR_DIMENSIONS)


def load_test_vectors():
    return load_file(FILEPATH_TEST_IMAGES_NAME, FILEPATH_TEST_IMAGES_VECTORS, TEST_NUM_VECTORS, TEST_VECTOR_DIMENSIONS)


def load_captions(file_captions):
    assert os.path.isfile(file_captions), "file doesnt exists " + file_captions
    return [line.strip().split("\t") for line in open(file_captions, encoding='utf-8')]


if __name__ == '__main__':
    (train_names, train_vectors) = load_train_vectors()
    (test_names, test_vectors) = load_test_vectors()

    df_train_names = pd.DataFrame(train_names, columns=['filename'])

    train_captions = load_captions(FILEPATH_TRAIN_CAPTION)
    test_captions = load_captions(FILEPATH_TEST_CAPTION)

    df_train_captions = pd.DataFrame(train_captions, columns=['filename', 'description'])
    df_test_captions = pd.DataFrame(test_captions, columns=['filename', 'description'])
    df_train_captions_grouped_by_description = df_train_captions.groupby('filename')['description'].apply(
        ' '.join).reset_index()
    df_test_captions_grouped_by_description = df_test_captions.groupby('filename')['description'].apply(
        ' '.join).reset_index()

    texts_train = df_train_captions_grouped_by_description['description']

    vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 3), max_df=0.9, min_df=0.002, norm=None)
    vectorizer.fit(texts_train)
    X_train = vectorizer.transform(texts_train)
    print("Training: {}".format(X_train.shape))

