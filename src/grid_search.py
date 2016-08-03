from __future__ import division
import pandas as pd
import datetime
from util import connect_db, create_features
from sklearn.cross_validation import train_test_split
import pickle
from sklearn import grid_search
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier

'''
short_desc_init
short_desc_init
accuracy: 0.591123473213, alpha: 1.0, max_features: 20000
accuracy: 0.592776232515, alpha: 0.5, max_features: 20000*
accuracy: 0.590800983593, alpha: 0.1, max_features: 20000
accuracy: 0.584149635184, alpha: 1e-09, max_features: 20000
accuracy: 0.58003789253, alpha: 0.0, max_features: 20000
accuracy: 0.592574676503, alpha: 1.0, max_features: 10000**
accuracy: 0.591244406821, alpha: 0.5, max_features: 10000
accuracy: 0.589591647519, alpha: 0.1, max_features: 10000
accuracy: 0.586729552143, alpha: 1e-09, max_features: 10000
accuracy: 0.586165195308, alpha: 0.0, max_features: 10000
accuracy: 0.58781795461, alpha: 1.0, max_features: 5000
accuracy: 0.588583867457, alpha: 0.5, max_features: 5000
accuracy: 0.587576087395, alpha: 0.1, max_features: 5000
accuracy: 0.58652799613, alpha: 1e-09, max_features: 5000
accuracy: 0.58652799613, alpha: 0.0, max_features: 5000

desc_init
accuracy: 0.539646067642, alpha: 1.0, max_features: 30000
accuracy: 0.542266295804, alpha: 0.5, max_features: 30000
accuracy: 0.541863183779, alpha: 0.1, max_features: 30000
accuracy: 0.545531503205, alpha: 1e-09, max_features: 30000***
accuracy: 0.545370258395, alpha: 0.0, max_features: 30000
accuracy: 0.538073930745, alpha: 1.0, max_features: 20000
accuracy: 0.540936026122, alpha: 0.5, max_features: 20000
accuracy: 0.540371669287, alpha: 0.1, max_features: 20000
accuracy: 0.543919055105, alpha: 1e-09, max_features: 20000
accuracy: 0.543959366308, alpha: 0.0, max_features: 20000
accuracy: 0.535090901762, alpha: 1.0, max_features: 10000
accuracy: 0.536582416253, alpha: 0.5, max_features: 10000
accuracy: 0.536783972266, alpha: 0.1, max_features: 10000
accuracy: 0.538356109163, alpha: 1e-09, max_features: 10000
accuracy: 0.538718909985, alpha: 0.0, max_features: 10000
'''


def vectorize_features(X, train):
    '''Copy pasted from bug_pipeline, blah...'''
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

if __name__ == '__main__':

    pd.options.mode.chained_assignment = None  # turn off warnings

    # Get data from database
    print '{}: connecting to database'.format(datetime.datetime.now())
    conn = connect_db()

    print '{}: loading data from database'.format(datetime.datetime.now())
    col_list = '''
        assigned_to_init, cc_init,
        product_init, version_init,
        component_init, op_sys_init, reporter_bug_cnt,
        desc_init, short_desc_init,
        priority_final, severity_final
        '''
    df_original = pd.read_sql_query(
        'select {} from final limit 100'.format(col_list), con=conn)

    for target in ['severity_final']:
        df = df_original.copy(deep=True)

        # Feature engineering
        print '{}: feature engineering {}'.format(datetime.datetime.now(), target)
        df = create_features(df, target=target)

        y_all = df.pop(target)
        X_all = df

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42)

    # Grid search NLP models
    for textual_data in []:  # 'short_desc_init', 'desc_init'
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

    # Grid search non-NLP model
    op_sys_vect = CountVectorizer()
    component_vect = CountVectorizer()
    X_train.drop(['short_desc_init', 'desc_init'], axis=1, inplace=True)
    X_test.drop(['short_desc_init', 'desc_init'], axis=1, inplace=True)
    X_train = vectorize_features(X_train, train=True)
    X_test = vectorize_features(X_test, train=False)
    for loss in ['deviance']: # 'deviance', 'exponential'
        for learning_rate in [1.0]:
            for n_estimators in [100]:
                for subsample in [1.0]:
                    feat_gb_model = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate,
                                                               n_estimators=n_estimators, subsample=subsample,
                                                               max_depth=3, init=None,
                                                               random_state=None, max_features=None,
                                                               verbose=0, max_leaf_nodes=None, warm_start=False)
                    feat_gb_model.fit(X_train, y_train)

                    print 'accuracy: {}, loss: {}, learning_rate: {}, n_estimators: {}, subsample: {}'.format(
                        feat_gb_model.score(X_test, y_test),
                        loss, learning_rate, n_estimators, subsample)
