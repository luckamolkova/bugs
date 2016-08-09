from __future__ import division
import pandas as pd
import numpy as np
import datetime
import random
from util import connect_db
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from scipy import spatial
from pathos.multiprocessing import ProcessingPool


class DuplicatePipeline(object):

    def __init__(self):
        """Initializes DuplicatePipeline."""

        self.model = GradientBoostingClassifier()
        self.tfidf_vect = TfidfVectorizer(input='content', lowercase=True, tokenizer=None,
                                          stop_words='english', use_idf=True,
                                          max_features=15000, ngram_range=(1, 2))

    def fit(self, X_dupl, y_dupl):
        """Fits the model.

        Args:
            X_dupl (dataframe): Dataframe with duplicate bug reports.
            y_dupl (dataframe): Dataframe with labels for X_dupl (1 means duplicate).

        Returns:
            None.
        """
        print '{}: generating non duplicates'.format(datetime.datetime.now())
        X_non_dupl = self.generate_non_duplicates(X_dupl)
        y_non_dupl = pd.Series(np.zeros(X_non_dupl.shape[0]), name='duplicate')

        X_all = pd.concat([X_dupl, X_non_dupl], axis=0)
        y_all = pd.concat([y_dupl, y_non_dupl], axis=0)

        print '{}: calculating distances'.format(datetime.datetime.now())
        X_distances = self.calculate_distances(X_all, train=True)

        print '{}: training model'.format(datetime.datetime.now())
        self.model.fit(X_distances, y_all)

        # print '{}: pickling model'.format(datetime.datetime.now())
        # pickle_path = '../data/duplicate_model.pkl'
        # with open(pickle_path, 'w') as f:
        #     pickle.dump(model, f)
        return

    def predict(self, X, topn=3):
        """Returns top n most likely duplicates for single observation.

        Args:
            X (dataframe): Bug report to check whether it is a duplicate.
            topn (integer): Number of best matches to return.

        Returns:
            dataframe: Top n best matches.
        """
        X_candidates_tmp = self.get_candidates(X.iloc[0]['product_init'], X.iloc[0]['opening'])

        # cross join bug report with all candidates
        X['tmp_key'] = 1
        X_candidates_tmp['tmp_key'] = 1
        X_candidates = pd.merge(X, X_candidates_tmp, on='tmp_key')
        X_candidates.drop(['tmp_key'], axis=1, inplace=True)

        X_candidates_distances = self.calculate_distances(
            X_candidates, train=False)
        probas = self.model.predict_proba(X_candidates_distances)[
            :, int(self.model.classes_[np.argmax(self.model.classes_)])]

        pos_of_topn = np.argsort(probas)[-topn:][::-1]
        return X_candidates.iloc[pos_of_topn]

    def evaluate(self, X_test):
        """Evaluates the model in parallel, stores result in db."""

        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS duplicate_eval (
            id            bigint NOT NULL,
            dupl_pos      int,
            candidates    int
            );
        """)
        conn.commit()
        conn.close()

        results = ProcessingPool().map(self.eval_single_dupl, X_test.iterrows())

        return results

    def generate_non_duplicates(self, X_dupl):
        """Generates non-duplicate records for given bug report.

        Args:
            X_dupl: Bug reports that are duplicates.

        Returns:
            dataframe: To each observation on the input attach info about 7 non-duplicates.
        """
        conn = connect_db()
        df_all_ids = pd.read_sql_query('''
            SELECT id, short_desc_init, desc_init, product_init, component_init, reporter, op_sys_init
            FROM final
            WHERE product_init='firefox'
            ''', con=conn)
        conn.close()

        duplicates = set(zip(X_dupl['id'], X_dupl['duplicate_of_id']))
        X_non_dupl = []
        for rownum, row in X_dupl.iterrows():
            for i in xrange(7):
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
                    row['reporter'],
                    row['op_sys_init'],
                    r['short_desc_init'],
                    r['desc_init'],
                    r['product_init'],
                    r['component_init'],
                    r['reporter'],
                    r['op_sys_init'],
                ])
        X_non_duplicates = pd.DataFrame(X_non_dupl)
        X_non_duplicates.columns = X_dupl.columns
        return X_non_duplicates

    def get_candidates(self, X_product_init, X_opening):
        """Gets potential candidates that could be duplicates from database.

        Args:
            X_product_init: Product of the bug report.
            X_opening: Timestamp when the bug report was created.

        TODO:
            Make it work for bug report not in db (unknown id).
        """
        conn = connect_db()
        # query = '''
        #     SELECT
        #         f.id
        #         , dof.id AS duplicate_of_id
        #         , f.short_desc_init
        #         , f.desc_init
        #         , f.product_init
        #         , f.component_init
        #         , f.reporter
        #         , f.op_sys_init
        #         , dof.short_desc_init AS dof_short_desc_init
        #         , dof.desc_init AS dof_desc_init
        #         , dof.product_init AS dof_product_init
        #         , dof.component_init AS dof_component_init
        #         , dof.reporter AS dof_reporter
        #         , dof.op_sys_init AS dof_op_sys_init
        #     FROM final f
        #         CROSS JOIN final dof
        #     WHERE 1 = 1
        #         AND f.id = {}
        #         -- original should be open when duplicate is created
        #         AND f.opening > dof.opening
        #         AND f.opening < dof.closing
        #         AND f.product_init = dof.product_init
        #     --ORDER BY dof.id
        #     --LIMIT 10
        # '''.format(X_id)

        query = '''
            SELECT
                dof.id AS duplicate_of_id
                , dof.short_desc_init AS dof_short_desc_init
                , dof.desc_init AS dof_desc_init
                , dof.product_init AS dof_product_init
                , dof.component_init AS dof_component_init
                , dof.reporter AS dof_reporter
                , dof.op_sys_init AS dof_op_sys_init
            FROM final dof
            WHERE 1 = 1
                -- original should be open when duplicate is created
                AND '{}' > dof.opening
                AND '{}' < dof.closing
                -- original and duplicate assumed to be in the same product
                AND '{}' = dof.product_init
            --ORDER BY dof.id
            --LIMIT 10
        '''.format(X_opening, X_opening, X_product_init)

        result = pd.read_sql_query(query, con=conn)
        conn.close()
        return result

    def calculate_distances(self, X_all, train=False):
        """Calculates distances between pairs of bug reports.

        Args:
            X_all (dataframe): Pairs of bug reports on each row.
            train (boolean): Whether it is train data (True) or not (False).

        Returns:
            dataframe: Distance matrix.
        """
        X = X_all.copy(deep=True)

        X['same_component'] = (X['component_init'] == X[
                               'dof_component_init']).astype(int)
        X['same_reporter'] = (X['reporter'] == X[
            'dof_reporter']).astype(int)
        X['same_op_sys'] = (X['op_sys_init'] == X[
            'dof_op_sys_init']).astype(int)

        if train:
            self.tfidf_vect.fit(X['desc_init'])

        X['desc_dist'] = 0
        X['short_desc_dist'] = 0
        for rownum, row in X.iterrows():
            row['desc_dist'] = spatial.distance.cosine(
                self.tfidf_vect.transform([row['desc_init']]).toarray()[0],
                self.tfidf_vect.transform([row['dof_desc_init']]).toarray()[0])
            row['short_desc_dist'] = spatial.distance.cosine(
                self.tfidf_vect.transform([row['short_desc_init']]).toarray()[0],
                self.tfidf_vect.transform([row['dof_short_desc_init']]).toarray()[0])
        return X[['same_component', 'desc_dist', 'short_desc_dist', 'same_reporter', 'same_op_sys']]

    def eval_single_dupl(self, X):
        """Returns the position of actual duplicate in search results.

        Args:
            X (dataframe): Bug report in database that is a duplicate.

        Returns:
            tuple (integer, integer): Position of actual duplicate, number of candidates.
        """
        conn = connect_db()
        cur = conn.cursor()
        query = '''
            SELECT
                f.id
                , f.short_desc_init
                , f.desc_init
                , f.product_init
                , f.component_init
                , f.reporter
                , f.op_sys_init
                , f.opening
            FROM final f
            WHERE f.id = {}
        '''.format(X[1]['id'])

        X_info = pd.read_sql_query(query, con=conn)
        X_candidates_tmp = self.get_candidates(X_info.iloc[0]['product_init'], X_info.iloc[0]['opening'])

        # cross join bug report with all candidates
        X_info['tmp_key'] = 1
        X_candidates_tmp['tmp_key'] = 1
        X_candidates = pd.merge(X_info, X_candidates_tmp, on='tmp_key')
        X_candidates.drop(['tmp_key'], axis=1, inplace=True)

        index_of_actual_duplicate = X_candidates[X_candidates[
            'duplicate_of_id'] == X[1]['duplicate_of_id']].index[0]

        X_candidates_distances = self.calculate_distances(
            X_candidates, train=False)
        probas = self.model.predict_proba(X_candidates_distances)[
            :, int(self.model.classes_[np.argmax(self.model.classes_)])]

        pos_of_actual_duplicate = np.argsort(
            probas)[::-1].tolist().index(index_of_actual_duplicate)

        cur.execute("""
            INSERT INTO duplicate_eval3
            VALUES (%s, %s, %s)
            """,
                    [X[1]['id'], pos_of_actual_duplicate, len(probas)]
                    )
        conn.commit()
        conn.close()

        print '{}: positioned as {} of {}'.format(datetime.datetime.now(), pos_of_actual_duplicate, len(probas))
        return (pos_of_actual_duplicate, len(probas))
