from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import datetime
import pickle
from collections import Counter
from nltk.corpus import words
from sklearn.feature_extraction.text import TfidfVectorizer


def get_nltk_vocabulary():
    '''Returns nltk vocabulary 'words' as numpy array.

    Returns:
        numpy array with unique words in the vocabulary.
    '''
    word_list = words.words()
    return np.unique([x.lower() for x in word_list])


def reduce_vocabulary(word_list, X):
    '''Reduces unique list of words on the input to words that appear in given dataset.

    Args:
        wordlist (array): Unique list of words.
        X (iterable): Iterable which yields either str, unicode or file objects.

    Returns:
        unique subset of input words that appear in given text
    '''
    vectorizer = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                 stop_words='english', use_idf=True, vocabulary=word_list,
                                 token_pattern='(?u)[a-zA-Z]+')
    tfidf = vectorizer.fit_transform(X)

    (_, nonzero_cols) = tfidf.nonzero()
    c = Counter(nonzero_cols)
    word_list_reduced = word_list[c.keys()]
    return word_list_reduced

if __name__ == '__main__':

    print '{}: loading pickled dataset'.format(datetime.datetime.now())
    df = pd.read_pickle('../data/df.pkl')

    print '{}: doing train test split'.format(datetime.datetime.now())
    X_all = df['desc_init']
    y_all = df['duration_bin']

    X, X_test, y, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42)

    print '{}: loading nltk vocabulary'.format(datetime.datetime.now())
    word_list = get_nltk_vocabulary()

    print '{}: reducing nltk vocabulary'.format(datetime.datetime.now())
    word_list_reduced = reduce_vocabulary(word_list, X)

    print '{}: pickling data'.format(datetime.datetime.now())
    pickle_path = '../data/vocabulary.pkl'
    with open(pickle_path, 'w') as f:
        pickle.dump(word_list_reduced, f)

    print '{}: data pickled as {}'.format(datetime.datetime.now(), pickle_path)
