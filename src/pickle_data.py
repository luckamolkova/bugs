from __future__ import division
from sqlalchemy import create_engine
import pandas as pd
import datetime


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
    df['duration'] = df['closing'] - df['opening']
    df['duration_days'] = df['duration'].apply(lambda x: float(x.days))
    df['duration_bin'] = pd.cut(df['duration_days'], bins=[-1, 1, 7, 30, 365, 10000], right=True,
                                labels=False)
    # labels=['day','week','month','year','more'])
    # is there assignee
    df['assigned_to_init_bool'] = df['assigned_to_init'].map(
        lambda x: 0 if x == '' else 1)

    # bug_status to int
    bug_status_map = dict({'new': 1, 'unconfirmed': 2, 'assigned': 3,
                           'resolved': 4, 'verified': 5, 'closed': 6, 'reopened': 7})
    df['bug_status_init'] = df['bug_status_init'].map(
        lambda x: bug_status_map[x] if x in bug_status_map.keys() else 0)
    df['bug_status_final'] = df['bug_status_final'].map(
        lambda x: bug_status_map[x] if x in bug_status_map.keys() else 0)

    # count number of initially cced
    df['cc_init_cnt'] = df['cc_init'].map(lambda x: x.count('@'))

    # priority to int
    priority_map = dict({'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4, 'p5': 5})
    df['priority_init'] = df['priority_init'].map(
        lambda x: priority_map[x] if x in priority_map.keys() else 0)
    df['priority_final'] = df['priority_final'].map(
        lambda x: priority_map[x] if x in priority_map.keys() else 0)

    # only keep top products
    product_map = dict({'core': 1, 'firefox': 2, 'thunderbird': 3,
                        'bugzilla': 4, 'browser': 5, 'webtools': 6, 'psm': 7})
    df['top_product_init'] = df['product_init'].map(
        lambda x: product_map[x] if x in product_map.keys() else 0)
    df['top_product_final'] = df['product_final'].map(
        lambda x: product_map[x] if x in product_map.keys() else 0)

    # severity to int
    severity_map = dict({'trivial': 1, 'minor': 2, 'normal': 3,
                         'major': 4, 'critical': 5, 'blocker': 6})
    df['severity_init'] = df['severity_init'].map(
        lambda x: severity_map[x] if x in severity_map.keys() else 0)
    df['severity_final'] = df['severity_final'].map(
        lambda x: severity_map[x] if x in severity_map.keys() else 0)

    # version to int
    version_map = dict({'trunk': 1, 'unspecified': 2, 'other': 3,
                        'other branch': 4, '2.0 branch': 5, '1.0 branch': 6})
    df['version_init'] = df['version_init'].map(
        lambda x: version_map[x] if x in version_map.keys() else 0)
    df['version_final'] = df['version_final'].map(
        lambda x: version_map[x] if x in version_map.keys() else 0)

    # short_desc_init_wordcnt
    df['short_desc_init_wordcnt'] = df[
        'short_desc_init'].map(lambda x: len(x.split()))

    # desc_wordcnt
    df['desc_init_wordcnt'] = df['desc_init'].map(lambda x: len(x.split()))

    return df

if __name__ == '__main__':

    pd.options.mode.chained_assignment = None #turn off warnings

    print '{}: connecting to database'.format(datetime.datetime.now())
    engine = create_engine('postgresql://lucka@localhost:5432/bugs')

    print '{}: loading data from database'.format(datetime.datetime.now())
    df = pd.read_sql_query('select * from final', con=engine)

    print '{}: preparing data'.format(datetime.datetime.now())
    df = prepare_data(df)

    print '{}: pickling data'.format(datetime.datetime.now())
    pickle_path = '../data/df.pkl'
    df.to_pickle(pickle_path)

    print '{}: data pickled as {}'.format(datetime.datetime.now(), pickle_path)
