from __future__ import division
import pandas as pd
import datetime
import pickle
from util import get_data
from bug_pipeline import BugPipeline

if __name__ == '__main__':

    pd.options.mode.chained_assignment = None  # turn off warnings

    targets=['severity_final', 'priority_final']

    # Fit and pickle pipeline(s)
    for target in targets:
        X, X_test, y, y_test = get_data(limit=None, target=target) # set limit=None to get everything
        pipeline = BugPipeline()
        print '{}: fitting pipeline for {}'.format(datetime.datetime.now(), target)
        pipeline.fit(X, y)
        print '{}: predicting for {}'.format(datetime.datetime.now(), target)
        class_report = pipeline.evaluate(X_test, y_test)
        print class_report

        pickle_path = '../data/{}_pipeline.pkl'.format(target)
        with open(pickle_path, 'w') as f:
            pickle.dump(pipeline, f)