from __future__ import division
import pandas as pd
import numpy as np
import datetime
import pickle
from util import get_data
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier


def vectorize_features(X, train):
    """Copy pasted from bug_pipeline, yeah, I know..."""
    global op_sys_vect, component_vect
    if train:
        # vectorize component
        component_value_cnts = X['component_init'].value_counts()
        X['component_init'] = X['component_init'].map(
            lambda x: x if component_value_cnts[x] > 100 else 'unimportant')

        component = component_vect.fit_transform(
            X.pop('component_init'))
        X = pd.concat([X.reset_index(drop=True),
                       pd.DataFrame(component.toarray(), columns=[
                                    'component_' + str(x) for x in xrange(component.shape[1])]).reset_index(drop=True)
                       ], axis=1, join='inner')

        # vectorize op_sys
        op_sys = op_sys_vect.fit_transform(X.pop('op_sys_init'))
        X = pd.concat([X.reset_index(drop=True),
                       pd.DataFrame(op_sys.toarray(), columns=[
                                    'op_sys_' + str(x) for x in xrange(op_sys.shape[1])]).reset_index(drop=True)
                       ], axis=1, join='inner')
    else:
        # vectorize component
        component = component_vect.transform(X.pop('component_init'))
        X = pd.concat([X.reset_index(drop=True),
                       pd.DataFrame(component.toarray(), columns=[
                                    'component_' + str(x) for x in xrange(component.shape[1])]).reset_index(drop=True)
                       ], axis=1, join='inner')

        # vectorize op_sys
        op_sys = op_sys_vect.transform(X.pop('op_sys_init'))
        X = pd.concat([X.reset_index(drop=True),
                       pd.DataFrame(op_sys.toarray(), columns=[
                                    'op_sys_' + str(x) for x in xrange(op_sys.shape[1])]).reset_index(drop=True)
                       ], axis=1, join='inner')

    return X


def grid_search_nlp(X_train, X_test, y_train, y_test, textual_data='desc_init'):
    """Grid search TfIdf vectorizer and Multinomial NB for best accuracy on text data."""
    print textual_data
    for max_features in [30000, 20000, 10000]:
        # initiate vectorizer as per grid params
        desc_vect = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                    stop_words='english', use_idf=True,
                                    max_features=max_features, ngram_range=(1, 3))
        desc_tfidf_train = desc_vect.fit_transform(X_train[textual_data])
        desc_tfidf_test = desc_vect.transform(X_test[textual_data])

        for alpha in [1.0, 0.5, 0.1, 1e-09, 0.0]:
            # initiate model as per grid params
            desc_nb_model = MultinomialNB(alpha=alpha)

            desc_nb_model.fit(desc_tfidf_train, y_train)

            print 'accuracy: {}, alpha: {}, max_features: {}'.format(desc_nb_model.score(desc_tfidf_test, y_test), alpha, max_features)


def grid_search_features(X_train, X_test, y_train, y_test):
    """Grid search Gradient Boosting classifier for best accuracy on features."""
    X_train.drop(['short_desc_init', 'desc_init'], axis=1, inplace=True)
    X_test.drop(['short_desc_init', 'desc_init'], axis=1, inplace=True)
    X_train = vectorize_features(X_train, train=True)
    X_test = vectorize_features(X_test, train=False)
    for learning_rate in [1.0]:
        for n_estimators in [200, 100, 50]:
            for subsample in [1.0, 0.9]:
                feat_gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=learning_rate,
                                                           n_estimators=n_estimators, subsample=subsample,
                                                           max_depth=3, init=None,
                                                           random_state=None, max_features=None,
                                                           verbose=0, max_leaf_nodes=None, warm_start=False)
                feat_gb_model.fit(X_train, y_train)

                print 'accuracy: {}, learning_rate: {}, n_estimators: {}, subsample: {}'.format(
                    feat_gb_model.score(X_test, y_test),
                    learning_rate, n_estimators, subsample)


def grid_search_linear_stacking():
    """Grid search Gradient Boosting classifier for best accuracy on meta-features."""
    X_ensem = pickle.load(open('../data/X_ensem.pkl', 'rb'))
    y_ensem = pickle.load(open('../data/y_ensem.pkl', 'rb'))

    X_train, X_test, y_train, y_test = train_test_split(
        X_ensem, y_ensem, test_size=0.2, random_state=42)

    for learning_rate in [0.01]:
        for n_estimators in [200]:
            for subsample in [1.0]:
                ensem_gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=learning_rate,
                                                            n_estimators=n_estimators, subsample=subsample,
                                                            max_depth=4, init=None,
                                                            random_state=42, max_features=None,
                                                            verbose=0, max_leaf_nodes=None, warm_start=False)
                ensem_gb_model.fit(X_train, y_train)

                print 'accuracy: {}, learning_rate: {}, n_estimators: {}, subsample: {}'.format(
                    ensem_gb_model.score(X_test, y_test),
                    learning_rate, n_estimators, subsample)

if __name__ == '__main__':

    pd.options.mode.chained_assignment = None  # turn off warnings

    op_sys_vect = CountVectorizer()
    component_vect = CountVectorizer()

    X_train, X_test, y_train, y_test = get_data(limit=100)

    grid_search_nlp(X_train, X_test, y_train, y_test, textual_data='desc_init')
    # grid_search_nlp(X_train, X_test, y_train, y_test, textual_data='short_desc_init')
    # grid_search_features(X_train, X_test, y_train, y_test)

    # grid_search_linear_stacking()
