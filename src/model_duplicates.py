from __future__ import division
import pandas as pd
import numpy as np
import datetime
import pickle
import random
from util import connect_db
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from scipy import spatial
from pathos.multiprocessing import ProcessingPool


def create_tfidf_vect(X, column='desc_init'):
    tfidf_vect = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                 stop_words='english', use_idf=True,
                                 max_features=30, ngram_range=(1, 3))

    tfidf_vect.fit(X[column])

    pickle_path = '../data/duplicate_{}_vectorizer.pkl'.format(column)
    with open(pickle_path, 'w') as f:
        pickle.dump(tfidf_vect, f)
    return tfidf_vect


def generate_non_duplicates(X_dupl, conn):
    df_all_ids = pd.read_sql_query(
        'SELECT id, short_desc_init, desc_init, product_init, component_init FROM final', con=conn)
    duplicates = set(zip(X_dupl['id'], X_dupl['duplicate_of_id']))
    X_non_dupl = []
    for rownum, row in X_dupl.iterrows():
        while True:
            rownum = random.randint(0, df_all_ids.shape[0] - 1)
            r = df_all_ids.iloc[rownum]
            if (row['id'], r['id']) not in duplicates and (r['id'], row['id']) not in duplicates:
                break
        X_non_dupl.append([
            row['id'],
            r['id'],
            row['short_desc_init'],
            row['desc_init'],
            row['product_init'],
            row['component_init'],
            r['short_desc_init'],
            r['desc_init'],
            r['product_init'],
            r['component_init'],
        ])
    X_non_duplicates = pd.DataFrame(X_non_dupl)
    X_non_duplicates.columns = X_dupl.columns
    return X_non_duplicates


def get_candidates(X_id):
    conn = connect_db()
    query = '''
        SELECT
            f.id
            , dof.id AS duplicate_of_id
            , f.short_desc_init
            , f.desc_init
            , f.product_init
            , f.component_init
            , dof.short_desc_init AS dof_short_desc_init
            , dof.desc_init AS dof_desc_init
            , dof.product_init AS dof_product_init
            , dof.component_init AS dof_component_init
        FROM final f
            CROSS JOIN final dof
        WHERE 1 = 1
            AND f.id = {}
            -- original should be open when duplicate is created
            AND f.opening > dof.opening
            AND f.opening < dof.closing
        --ORDER BY dof.id
        --LIMIT 10
    '''.format(X_id)

    result = pd.read_sql_query(query, con=conn)
    conn.close()
    return result


def calculate_distances(X_all, tfidf_vect=None, column='desc_init'):
    X = X_all.copy(deep=True)
    X['same_product'] = (X['product_init'] == X[
                         'dof_product_init']).astype(int)
    X['same_component'] = (X['component_init'] == X[
                           'dof_component_init']).astype(int)

    if not tfidf_vect:
        tfidf_vect = create_tfidf_vect(X, column=column)

    X['desc_dist'] = 0
    for rownum, row in X.iterrows():
        row['desc_dist'] = spatial.distance.cosine(
            tfidf_vect.transform([row['desc_init']]).toarray()[0],
            tfidf_vect.transform([row['dof_desc_init']]).toarray()[0])
    return X[['same_product', 'same_component', 'desc_dist']]


def eval_single_dupl(X):
    X_candidates = get_candidates(X[1]['id'])
    index_of_actual_duplicate = X_candidates[X_candidates[
        'duplicate_of_id'] == X[1]['duplicate_of_id']].index[0]

    X_candidates_distances = calculate_distances(
        X_candidates, tfidf_vect=tfidf_vect)
    probas = model.predict_proba(X_candidates_distances)[
        :, int(model.classes_[np.argmax(model.classes_)])]

    pos_of_actual_duplicate = np.argsort(
        probas)[::-1].tolist().index(index_of_actual_duplicate)

    return (pos_of_actual_duplicate, len(probas))

if __name__ == "__main__":
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
            , dof.short_desc_init AS dof_short_desc_init
            , dof.desc_init AS dof_desc_init
            , dof.product_init AS dof_product_init
            , dof.component_init AS dof_component_init
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
        --ORDER BY d.id
        LIMIT 10
    '''
    df_original = pd.read_sql_query(query, con=conn)
    df_original['duplicate'] = 1

    y_data = df_original.pop('duplicate')
    X_data = df_original

    X_dupl, X_dupl_test, y_dupl, y_dupl_test = train_test_split(
        X_data, y_data, test_size=0.25, random_state=42)

    # train model
    print '{}: generating non duplicates'.format(datetime.datetime.now())
    X_non_dupl = generate_non_duplicates(X_dupl, conn)
    y_non_dupl = pd.Series(np.zeros(X_non_dupl.shape[0]), name='duplicate')

    X_all = pd.concat([X_dupl, X_non_dupl], axis=0)
    y_all = pd.concat([y_dupl, y_non_dupl], axis=0)

    print '{}: calculation distances'.format(datetime.datetime.now())
    X_distances = calculate_distances(X_all)

    print '{}: training model'.format(datetime.datetime.now())
    model = GradientBoostingClassifier()
    model.fit(X_distances, y_all)

    print '{}: pickling model'.format(datetime.datetime.now())
    pickle_path = '../data/duplicate_model.pkl'
    with open(pickle_path, 'w') as f:
        pickle.dump(model, f)

    # evaluate model
    tfidf_vect = pickle.load(
        open('../data/duplicate_desc_init_vectorizer.pkl', 'rb'))
    print '{}: evaluating model'.format(datetime.datetime.now())

    results = ProcessingPool().map(eval_single_dupl, X_dupl_test.iterrows())

    print results