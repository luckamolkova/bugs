from __future__ import division
import pandas as pd
import numpy as np
import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
import pickle


class BugPipeline(object):

    def __init__(self):
        """Initializes BugPipeline."""

        # NLP description
        self.desc_vect = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                         stop_words='english', use_idf=True,
                                         max_features=30000, ngram_range=(1, 3))
        self.desc_nb_model = MultinomialNB(alpha=1e-09)

        # NLP short description
        self.short_desc_vect = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                               stop_words='english', use_idf=True,
                                               max_features=10000, ngram_range=(1, 3))
        self.short_desc_nb_model = MultinomialNB(alpha=1.0)

        # Non-textual features
        self.op_sys_vect = CountVectorizer()
        self.component_vect = CountVectorizer()
        self.feat_gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.5,
                                                        n_estimators=100, subsample=1.0,
                                                        max_depth=3, init=None,
                                                        random_state=None, max_features=None,
                                                        verbose=0, max_leaf_nodes=None, warm_start=False)

        # Linear stacking
        self.ensem_gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                                         n_estimators=100, subsample=1.0,
                                                         max_depth=3, init=None,
                                                         random_state=None, max_features=None,
                                                         verbose=0, max_leaf_nodes=None, warm_start=False)

    def fit(self, X_data, y_data):
        """Fits bug pipeline.

        Three classifier models in the first level are built:
        1) Multinomial Naive Bayes on short description (tfidf with 2,000 features)
        2) Multinomial Naive Bayes on long description (tfidf with 20,000 features)
        3) Gradient Boosting on other features

        Linear Stacking is used and another Gradient Boosting classifier
        runs using meta-features (probabilities predicted by first level models).

        Args:
            X_data (dataframe): Features.
            y_data (series): Target.

        Returns:
            None.
        """
        X = X_data.copy(deep=True)
        X.set_index([range(X.shape[0])], inplace=True)
        y = y_data.copy(deep=True)
        y.reset_index(drop=True, inplace=True)

        # Non-textual features
        print '{}: Non-textual features fit'.format(datetime.datetime.now())
        X = self.vectorize_features(X, train=True)
        fold = 0
        # KFold training needed for linear stacking
        kf = KFold(X.shape[0], n_folds=5, shuffle=False, random_state=42)
        for train_index, test_index in kf:
            print '{}: KFold training fold {}'.format(datetime.datetime.now(), fold)
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            # NLP description
            desc_tfidf = self.desc_vect.fit_transform(X_train.pop('desc_init'))
            self.desc_nb_model.fit(desc_tfidf, y_train)

            # NLP short description
            short_desc_tfidf = self.short_desc_vect.fit_transform(
                X_train.pop('short_desc_init'))
            self.short_desc_nb_model.fit(short_desc_tfidf, y_train)

            self.feat_gb_model.fit(X_train, y_train)

            # Collect data for linear stacking
            if fold > 0:
                X_ensem = np.vstack((X_ensem,
                                     np.hstack(
                                         (self.desc_nb_model.predict_proba(self.desc_vect.transform(X_test.pop('desc_init'))),
                                          self.short_desc_nb_model.predict_proba(
                                             self.short_desc_vect.transform(X_test.pop('short_desc_init'))),
                                             self.feat_gb_model.predict_proba(X_test)))))
                y_ensem = np.hstack((y_ensem, y_test.values))
            else:
                X_ensem = np.hstack(
                    (self.desc_nb_model.predict_proba(self.desc_vect.transform(X_test.pop('desc_init'))),
                     self.short_desc_nb_model.predict_proba(
                         self.short_desc_vect.transform(X_test.pop('short_desc_init'))),
                     self.feat_gb_model.predict_proba(X_test)))
                y_ensem = y_test.values
            fold += 1

        pickle_path = '../data/X_ensem.pkl'
        with open(pickle_path, 'w') as f:
            pickle.dump(X_ensem, f)

        pickle_path = '../data/y_ensem.pkl'
        with open(pickle_path, 'w') as f:
            pickle.dump(y, f)

        # Linear stacking
        print '{}: training ensemble'.format(datetime.datetime.now())
        self.ensem_gb_model.fit(X_ensem, y)

    def predict(self, X_data):
        """Predicts.

        Args:
            X_data (dataframe): Features.

        Returns:
            Return predictions.
        """
        X = X_data.copy(deep=True)

        # NLP description
        # print '{}: NLP description transform and predict'.format(datetime.datetime.now())
        desc_tfidf = self.desc_vect.transform(X.pop('desc_init'))
        desc_y_pred = self.desc_nb_model.predict(desc_tfidf)
        # print 'desc classification report: \n {}'.format(classification_report(y, desc_y_pred))

        # NLP short description
        # print '{}: NLP short description transform and predict'.format(datetime.datetime.now())
        short_desc_tfidf = self.short_desc_vect.transform(
            X.pop('short_desc_init'))
        short_desc_y_pred = self.short_desc_nb_model.predict(short_desc_tfidf)
        # print 'short desc classification report: \n {}'.format(classification_report(y, short_desc_y_pred))

        # Non-textual features
        # print '{}: Non-textual features transform and predict'.format(datetime.datetime.now())
        X = self.vectorize_features(X, train=False)

        feat_y_pred = self.feat_gb_model.predict(X)
        # print 'feature classification report: \n {}'.format(classification_report(y, feat_y_pred))

        # Linear stacking
        X_ensem = np.hstack(
            (self.desc_nb_model.predict_proba(desc_tfidf),
             self.short_desc_nb_model.predict_proba(short_desc_tfidf),
             self.feat_gb_model.predict_proba(X)))
        ensem_y_pred = self.ensem_gb_model.predict(X_ensem)

        return ensem_y_pred

    def evaluate(self, X, y):
        """Prints classification report for the final predictions.

        Args:
            X (dataframe): Features.
            y (dataframe): Target.

        Returns:
            Sklearn classification report.
        """
        y_pred = self.predict(X)
        return classification_report(y, y_pred)


    def vectorize_features(self, X, train):
        """Predicts and evaluates the predictions.

        Args:
            X (dataframe): Dataframe with `component_init` and `op_sys_init` columns.
            train (boolean): Whether it is train data (fit) or not (transform).

        Returns:
            dataframe: Dataframe with `component_init` and `op_sys_init` vectorized.

        TODO:
            This is kind of ugly and repetitive. Improve?
        """
        if train:
            # vectorize component
            component_value_cnts = X['component_init'].value_counts()
            X['component_init'] = X['component_init'].map(
                lambda x: x if component_value_cnts[x] > 100 else 'unimportant')

            component = self.component_vect.fit_transform(
                X.pop('component_init'))
            X = pd.concat([X.reset_index(drop=True),
                           pd.DataFrame(component.toarray(), columns=[
                                        'component_' + str(x) for x in xrange(component.shape[1])]).reset_index(drop=True)
                           ], axis=1, join='inner')

            # vectorize op_sys
            op_sys = self.op_sys_vect.fit_transform(X.pop('op_sys_init'))
            X = pd.concat([X.reset_index(drop=True),
                           pd.DataFrame(op_sys.toarray(), columns=[
                                        'op_sys_' + str(x) for x in xrange(op_sys.shape[1])]).reset_index(drop=True)
                           ], axis=1, join='inner')
        else:
            # vectorize component
            component = self.component_vect.transform(X.pop('component_init'))
            X = pd.concat([X.reset_index(drop=True),
                           pd.DataFrame(component.toarray(), columns=[
                                        'component_' + str(x) for x in xrange(component.shape[1])]).reset_index(drop=True)
                           ], axis=1, join='inner')

            # vectorize op_sys
            op_sys = self.op_sys_vect.transform(X.pop('op_sys_init'))
            X = pd.concat([X.reset_index(drop=True),
                           pd.DataFrame(op_sys.toarray(), columns=[
                                        'op_sys_' + str(x) for x in xrange(op_sys.shape[1])]).reset_index(drop=True)
                           ], axis=1, join='inner')

        return X
