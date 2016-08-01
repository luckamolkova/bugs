from __future__ import division
import pandas as pd
import datetime
from util import connect_db, create_features
from bug_pipeline import BugPipeline
from sklearn.cross_validation import train_test_split
import pickle

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
        ''' #priority_init, severity_init, bug_status_init
    df_original = pd.read_sql_query('select {} from final'.format(col_list), con=conn)

    # for target in ['priority_final', 'severity_final']:
    for target in ['priority_final']:
        df = df_original.copy(deep=True)

        # Feature engineering
        print '{}: feature engineering {}'.format(datetime.datetime.now(), target)
        df = create_features(df, target=target)

        y_all = df.pop(target)
        X_all = df

        X, X_test, y, y_test = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42)

        # Fit and pickle pipeline
        pipeline = BugPipeline()
        print '{}: fitting pipeline for {}'.format(datetime.datetime.now(), target)
        pipeline.fit(X, y)
        print '{}: predicting for {}'.format(datetime.datetime.now(), target)
        pipeline.predict(X_test, y_test)

        # pickle_path = '../data/{}_pipeline.pkl'.format(target)
        # with open(pickle_path, 'w') as f:
        #     pickle.dump(pipeline, f)
