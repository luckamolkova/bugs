from __future__ import division
import pandas as pd
import datetime
from util import connect_db, create_features
from severity_pipeline import SeverityPipeline
from sklearn.cross_validation import train_test_split
import pickle

if __name__ == '__main__':

    pd.options.mode.chained_assignment = None  # turn off warnings

    print '{}: connecting to database'.format(datetime.datetime.now())
    conn = connect_db()

    # Features
    print '{}: loading data from database'.format(datetime.datetime.now())
    col_list = '''
        assigned_to_init, bug_status_init, cc_init,
        priority_init, product_init, version_init,
        component_init, op_sys_init, reporter_bug_cnt,
        desc_init, short_desc_init,
        severity_final
        '''
    df = pd.read_sql_query('select {} from final'.format(col_list), con=conn)

    print '{}: feature engineering'.format(datetime.datetime.now())
    df = create_features(df)

    y_all = df.pop('severity_final')
    X_all = df

    X, X_test, y, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=42)

    # get pipeline
    pipeline = SeverityPipeline()
    pipeline.fit(X, y)
    pipeline.predict(X_test, y_test)

    pickle_path = '../data/severity_pipeline.pkl'
    with open(pickle_path, 'w') as f:
        pickle.dump(pipeline, f)