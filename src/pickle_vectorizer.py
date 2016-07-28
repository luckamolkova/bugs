from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


if __name__ == '__main__':
    df = pd.read_pickle('../data/df.pkl')
    word_list_reduced = pickle.load(open('../data/vocabulary.pkl', 'rb'))

    X_all = df['desc_init']
    y_all = df['duration_bin']

    X, X_test, y, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                 stop_words='english', use_idf=True, vocabulary=word_list_reduced,
                                 token_pattern='(?u)[a-zA-Z]+',)  # default token_pattern='(?u)\b\w\w+\b' '[a-zA-Z]+'
    tfidf = vectorizer.fit_transform(X)
    print tfidf.shape

    pickle_path = '../data/tfidf_vectorizer.pkl'
    with open(pickle_path, 'w') as f:
        pickle.dump(vectorizer, f)

    # PCA ends up being killed by OS (using too much memory?)
    # model = PCA(n_components=100)  # number of dimensions/topics required
    # model.fit(tfidf.toarray())
    # pca = model.transform(tfidf.toarray()[:100])

    # pickle_path = '../data/pca_model.pkl'
    # with open(pickle_path, 'w') as f:
    #     pickle.dump(model, f)

    # print sum(model.explained_variance_ratio_)
