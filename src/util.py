from __future__ import division
import psycopg2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def connect_db():
    '''Returns psycopg2 connection to PostgreSQL database.'''
    try:
        conn = psycopg2.connect("dbname='bugs' user='lucka' host='localhost'")
    except:
        print "Unable to connect to the database"
        exit(1)
    return conn

def create_empty_tables(conn, tables):
    '''Creates empty tables unless they already exist.

    Args:
        conn: Psycopg2 connection to PostgreSQL database.
        tables (list): List of tables to be created in addition to reports table.

    Returns:
        None.
    '''
    cur = conn.cursor()

    # create reports table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
        opening             timestamp,
        reporter            bigint,
        current_status      varchar(50),
        current_resolution  varchar(50),
        id                  bigint NOT NULL
        );
    """)
    conn.commit()

    # create other tables
    for table in tables:
        query = """
            CREATE TABLE IF NOT EXISTS {} (
            when_created    timestamp,
            what            text,
            who             bigint,
            id              bigint NOT NULL
            );
        """.format(table)
        cur.execute(query)
        conn.commit()

    return

def create_features(df, target):
    '''Take the dataframe and make it ready for models.

    Args:
        df (dataframe): Pandas dataframe with data.

    Returns:
        dataframe: modified input dataframe
    '''

    if target == 'priority_final':
        df.drop(['severity_final'], axis=1, inplace=True)
        # get rid of empty and -- (default)
        df = df[df['priority_final'] != '']
        df = df[df['priority_final'] != '--']

        # severity_vocabulary = ['other', 'trivial', 'minor', 'normal',
        #                        'major', 'critical', 'blocker']
        # df = one_hot(df, 'severity_init', severity_vocabulary)

    if target == 'severity_final':
        df.drop(['priority_final'], axis=1, inplace=True)
        # get rid of enhancements and normal (default)
        df = df[df['severity_final'] != 'enhancement']
        df = df[df['severity_final'] != 'normal']

        # priority_vocabulary = ['other', 'p1', 'p2', 'p3', 'p4', 'p5']
        # df = one_hot(df, 'priority_init', priority_vocabulary)

    # is there assignee
    df['assigned_to_init_bool'] = df.pop('assigned_to_init').map(
        lambda x: 0 if x == '' else 1)

    # count number of initially cced
    df['cc_init_cnt'] = df.pop('cc_init').map(lambda x: x.count('@'))

    # short_desc_init wordcount
    df['short_desc_init_wordcnt'] = df['short_desc_init'].map(lambda x: len(x.split()))

    # desc_wordcnt
    df['desc_init_wordcnt'] = df['desc_init'].map(lambda x: len(x.split()))

    # one hot encodings
    product_vocabulary = ['other', 'core', 'firefox', 'thunderbird',
                          'bugzilla', 'browser', 'webtools', 'psm']
    df = one_hot(df, 'product_init', product_vocabulary)

    version_vocabulary = ['other', 'trunk', 'unspecified',
                          'other branch', '2.0 branch', '1.0 branch']
    df = one_hot(df, 'version_init', version_vocabulary)

    return df


def one_hot(df, colname, vocabulary):
    '''Performs one hot encoding of specified column in a dataframe.

    Args:
        df (dataframe): Dataframe with a column to encode.
        colname (string): Name of column to perform one hot encoding on.
        vocabulary (list of strings): List of values to encode.

    Returns:
        dataframe: Original dataframe with the initial column replaced with
        new x columns (x is number of items in the vocabulary)
    '''
    cnt_vectorizer = CountVectorizer(vocabulary=vocabulary)
    data = cnt_vectorizer.fit_transform(df.pop(colname).map(
        lambda x: x if x in vocabulary else 'other'))
    colnames = [colname + '_' + x for x in vocabulary]
    df = pd.concat([
        df.reset_index(drop=True),
        pd.DataFrame(data.toarray(), columns=colnames).reset_index(drop=True)],
        axis=1, join='inner')
    return df
