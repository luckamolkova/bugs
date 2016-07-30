from __future__ import division
import pandas as pd
import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


class SeverityPipeline(object):

    def __init__(self):
        # NLP description
        self.desc_vect = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                         stop_words='english', use_idf=True,
                                         max_features=20000, ngram_range=(1, 3))
        self.desc_nb_model = MultinomialNB()

        # NLP short description
        self.short_desc_vect = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                               stop_words='english', use_idf=True,
                                               max_features=20000, ngram_range=(1, 3))
        self.short_desc_nb_model = MultinomialNB()

        # Non-textual features
        self.op_sys_vect = CountVectorizer()
        self.component_vect = CountVectorizer()
        self.feat_gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                                        n_estimators=100, subsample=1.0,
                                                        max_depth=3, init=None,
                                                        random_state=None, max_features=None,
                                                        verbose=0, max_leaf_nodes=None, warm_start=False)

    def fit(self, X, y):
        # NLP description
        print '{}: NLP description fit'.format(datetime.datetime.now())
        desc_tfidf = self.desc_vect.fit_transform(X.pop('desc_init'))
        self.desc_nb_model.fit(desc_tfidf, y)

        # # NLP short description
        # print '{}: NLP short description fit'.format(datetime.datetime.now())
        # short_desc_tfidf = self.short_desc_vect.fit_transform(
        #     X.pop('short_desc_init'))
        # self.short_desc_nb_model.fit(short_desc_tfidf, y)
        X.drop('short_desc_init', axis=1, inplace=True)

        # Non-textual features
        print '{}: Non-textual features fit'.format(datetime.datetime.now())
        # vectorize component
        component_value_cnts = X['component_init'].value_counts()
        X['component_init'] = X['component_init'].map(
            lambda x: x if component_value_cnts[x] > 100 else 'unimportant')

        component = self.component_vect.fit_transform(X.pop('component_init'))
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

        print X.shape, op_sys.shape, component.shape

        self.feat_gb_model.fit(X, y)

    def predict(self, X, y):
        # NLP description
        print '{}: NLP description transform and predict'.format(datetime.datetime.now())
        desc_tfidf = self.desc_vect.transform(X.pop('desc_init'))
        desc_y_pred = self.desc_nb_model.predict(desc_tfidf)
        print 'desc classification report: \n {}'.format(classification_report(y, desc_y_pred))

        # # NLP short description
        # print '{}: NLP short description transform and predict'.format(datetime.datetime.now())
        # short_desc_tfidf = self.short_desc_vect.transform(X.pop('short_desc_init'))
        # short_desc_y_pred = self.short_desc_nb_model.predict(short_desc_tfidf)
        # print 'short desc classification report: \n {}'.format(classification_report(y, short_desc_y_pred))
        X.drop('short_desc_init', axis=1, inplace=True)

        # Non-textual features
        print '{}: Non-textual features transform and predict'.format(datetime.datetime.now())
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

        feat_y_pred = self.feat_gb_model.predict(X)
        print 'feature classification report: \n {}'.format(classification_report(y, feat_y_pred))
