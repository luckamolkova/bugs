from __future__ import division
import pandas as pd
import numpy as np
import datetime
import pickle
import random
from util import connect_db
from sklearn.cross_validation import train_test_split
from duplicate_pipeline import DuplicatePipeline

if __name__ == '__main__':

    conn = connect_db()

    print '{}: getting data'.format(datetime.datetime.now())
    query = '''
        SELECT
            d.id
            , d.duplicate_of_id
            , f.short_desc_init
            , f.desc_init
            , f.product_init
            , f.component_init
            , f.reporter
            , f.op_sys_init
            , dof.short_desc_init AS dof_short_desc_init
            , dof.desc_init AS dof_desc_init
            , dof.product_init AS dof_product_init
            , dof.component_init AS dof_component_init
            , dof.reporter AS dof_reporter
            , dof.op_sys_init AS dof_op_sys_init
        FROM duplicates d
            JOIN final f
                ON d.id = f.id
            JOIN final dof
                ON d.duplicate_of_id = dof.id
        WHERE 1 = 1
            -- only bugs that are in db >> down from 90,000 to 60,000 rows
            AND d.duplicate_of_id IS NOT NULL
            -- original should be open when duplicate is created >> down to 40,000 rows
            AND f.opening > dof.opening
            AND f.opening < dof.closing
            -- only firefox product and candidates in the same product >> down to 10,000 rows
            AND f.product_init = 'firefox'
            AND f.product_init = dof.product_init
        --ORDER BY d.id
        --LIMIT 10
    '''
    df_original = pd.read_sql_query(query, con=conn)
    df_original['duplicate'] = 1

    y_data = df_original.pop('duplicate')
    X_data = df_original

    X_dupl, X_dupl_test, y_dupl, y_dupl_test = train_test_split(
        X_data, y_data, test_size=0.25, random_state=42)

    pipeline = DuplicatePipeline()
    print '{}: fitting pipeline for duplicates'.format(datetime.datetime.now())
    pipeline.fit(X_dupl, y_dupl)

    pickle_path = '../data/duplicate_pipeline.pkl'
    with open(pickle_path, 'w') as f:
        pickle.dump(pipeline, f)

    # pipeline.evaluate(X_dupl_test)
    conn.close()
