from __future__ import division
from sqlalchemy import create_engine
import pandas as pd
import scipy
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict


def prepare_data(df):
    '''Take the dataframe and make it ready for models.

    Args:
        df (dataframe): Pandas dataframe with data.

    Returns:
        dataframe: modified input dataframe
    '''

    # get rid of enhancements
    df = df[df['severity_init'] != 'enhancement']

    # calc resolution time (duration)
    df['duration_days'] = (df['closing'] - df['opening']).apply(lambda x: float(x.days))
    df.drop(['closing', 'opening'], axis=1, inplace=True)

    # is there assignee
    df['assigned_to_init_bool'] = df.pop('assigned_to_init').map(
        lambda x: 0 if x == '' else 1)

    # bug_status one hot
    bug_status_vocabulary = ['new', 'unconfirmed', 'assigned',
                             'resolved', 'verified', 'closed', 'reopened']
    df = one_hot(df, 'bug_status_init', bug_status_vocabulary)
    #df = one_hot(df, 'bug_status_final', bug_status_vocabulary)

    # count number of initially cced
    df['cc_init_cnt'] = df.pop('cc_init').map(lambda x: x.count('@'))

    # priority one hot
    priority_vocabulary = ['other', 'p1', 'p2', 'p3', 'p4', 'p5']
    df = one_hot(df, 'priority_init', priority_vocabulary)
    #df = one_hot(df, 'priority_final', priority_vocabulary)

    # product one hot (only keep top few products)
    product_vocabulary = ['other', 'core', 'firefox', 'thunderbird',
                          'bugzilla', 'browser', 'webtools', 'psm']
    df = one_hot(df, 'product_init', product_vocabulary)
    #df = one_hot(df, 'product_final', product_vocabulary)

    # severity one hot
    severity_vocabulary = ['other', 'trivial', 'minor', 'normal',
                           'major', 'critical', 'blocker']
    df = one_hot(df, 'severity_init', severity_vocabulary)
    #df = one_hot(df, 'severity_final', severity_vocabulary)

    # version one hot
    version_vocabulary = ['other', 'trunk', 'unspecified',
                          'other branch', '2.0 branch', '1.0 branch']
    df = one_hot(df, 'version_init', version_vocabulary)
    #df = one_hot(df, 'version_final', version_vocabulary)

    # short_desc_init_wordcnt
    df['short_desc_init_wordcnt'] = df.pop(
        'short_desc_init').map(lambda x: len(x.split()))

    # desc_wordcnt
    df['desc_init_wordcnt'] = df.pop('desc_init').map(lambda x: len(x.split()))

    return df


def one_hot(df, colname, vocabulary):
    cnt_vectorizer = CountVectorizer(vocabulary=vocabulary)
    data = cnt_vectorizer.fit_transform(df.pop(colname).map(
        lambda x: x if x in vocabulary else 'other'))
    colnames = [colname + '_' + x for x in vocabulary]
    df = pd.concat([
        df.reset_index(drop=True),
        pd.DataFrame(data.toarray(), columns=colnames).reset_index(drop=True)],
        axis=1, join='inner')
    return df

if __name__ == '__main__':

    pd.options.mode.chained_assignment = None  # turn off warnings

    print '{}: connecting to database'.format(datetime.datetime.now())
    engine = create_engine('postgresql://lucka@localhost:5432/bugs')

    print '{}: loading data from database'.format(datetime.datetime.now())
    col_list = '''
        assigned_to_init, bug_status_init, cc_init,
        priority_init, product_init, severity_init,
        version_init, short_desc_init, desc_init,
        opening, closing, severity_final,
        component_init, op_sys_init, reporter_bug_cnt
        ''' #resolution_final
    df = pd.read_sql_query('select {} from final'.format(col_list), con=engine)

    print '{}: preparing data'.format(datetime.datetime.now())
    df = prepare_data(df)

    print '{}: pickling data'.format(datetime.datetime.now())
    pickle_path = '../data/df_severity.pkl'
    df.to_pickle(pickle_path)

    # DURATION NLP
    # print '{}: loading nlp data from database'.format(datetime.datetime.now())
    # col_list = '''
    #     opening, closing, resolution_final,
    #     short_desc_init, desc_init
    #     '''
    # df = pd.read_sql_query('select {} from final'.format(col_list), con=engine)

    # print '{}: pickling nlp data'.format(datetime.datetime.now())
    # pickle_path = '../data/df_nlp.pkl'
    # df.to_pickle(pickle_path)

    # print '{}: nlp data pickled as {}'.format(datetime.datetime.now(), pickle_path)

    # SEVERITY NLP

    # print '{}: loading nlp data from database'.format(datetime.datetime.now())
    # col_list = '''
    #     severity_final,
    #     short_desc_init, desc_init
    #     '''
    # df = pd.read_sql_query('select {} from final'.format(col_list), con=engine)

    # print '{}: pickling nlp data'.format(datetime.datetime.now())
    # pickle_path = '../data/df_nlp_severity.pkl'
    # df.to_pickle(pickle_path)

    # print '{}: nlp data pickled as {}'.format(datetime.datetime.now(), pickle_path)
