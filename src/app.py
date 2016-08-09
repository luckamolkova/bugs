from __future__ import division
from flask import Flask, render_template, request
from random import random
import pandas as pd
import numpy as np
import pickle
from util import create_features
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model', methods=['GET', 'POST'])
def model():
    """Returns html with predictions (priority, severity and potential duplicates)."""

    # format data for severity and priority predictions
    X = pd.DataFrame([[
        request.args.get('assignee'),
        request.args.get('cc'),
        request.args.get('product'),
        'trunk',  # version
        request.args.get('component'),
        request.args.get('op_sys'),  # op_sys
        7,  # reporter_bug_cnt
        request.args.get('desc'),
        request.args.get('short_desc'),
        '-', '-']])  # priority_final and severity_final left blank, to be predicted

    X.columns = ['assigned_to_init', 'cc_init', 'product_init',
                 'version_init', 'component_init', 'op_sys_init',
                 'reporter_bug_cnt', 'desc_init', 'short_desc_init',
                 'severity_final', 'priority_final']

    X_severity = create_features(X.copy(deep=True), target='severity_final')
    X_severity.pop('severity_final')

    X_priority = create_features(X.copy(deep=True), target='priority_final')
    X_priority.pop('priority_final')

    # format string with priority and severity predictions
    html1 = '''
        <h3>Predicted priority:  {} </h3>
        <h3>Predicted severity: {} </h3>
    '''.format(priority_model.predict(X_priority)[0].upper(), severity_model.predict(X_severity)[0])

    # format data for duplicates prediction
    X_dupl = pd.DataFrame([[
        request.args.get('short_desc'),
        request.args.get('desc'),
        request.args.get('product'),
        request.args.get('component'),
        '12345',  # reporter
        request.args.get('op_sys'),
        '2012-12-10 21:05:10']])  # opening

    X_dupl.columns = ['short_desc_init', 'desc_init', 'product_init',
                      'component_init', 'reporter', 'op_sys_init', 'opening']

    duplicates = duplicate_model.predict(X_dupl)

    # format string with duplicate predictions
    if duplicates.shape[0] >= 3:
        html2 = '''
            <h3>Potential duplicates: </h3>
            <ul>
                <li><a style="color: #fff;" href="https://bugzilla.mozilla.org/show_bug.cgi?id={dof0}" target="new">{dof0}</a> <dd>{desc0}</dd></li>
                <li><a style="color: #fff;" href="https://bugzilla.mozilla.org/show_bug.cgi?id={dof1}" target="new">{dof1}</a> <dd>{desc1}</dd></li>
                <li><a style="color: #fff;" href="https://bugzilla.mozilla.org/show_bug.cgi?id={dof2}" target="new">{dof2}</a> <dd>{desc2}</dd></li>
            </ul>
        '''.format(
            dof0 = duplicates.iloc[0]['duplicate_of_id'],
            desc0 = duplicates.iloc[0]['dof_short_desc_init'],
            dof1 = duplicates.iloc[1]['duplicate_of_id'],
            desc1 = duplicates.iloc[1]['dof_short_desc_init'],
            dof2 = duplicates.iloc[2]['duplicate_of_id'],
            desc2 = duplicates.iloc[2]['dof_short_desc_init']
        )
    else:
        html2 = '''
            <h3>Potential duplicates: </h3>
            <ul>
                <li>None found. Only bug reports in the same product that were open as of 2012-12-10
                are considered by this search. Try produduct Firefox, Bugzilla or Thunderbird.</li>
            </ul>
        '''

    return str(html1) + str(html2)


if __name__ == '__main__':

    # load models
    severity_model = pickle.load(
        open('../data/severity_final_pipeline.pkl', 'r'))
    priority_model = pickle.load(
        open('../data/priority_final_pipeline.pkl', 'r'))
    duplicate_model = pickle.load(
        open('../data/duplicate_pipeline.pkl', 'r'))

    # start Flask app
    app.run(host='0.0.0.0', port=5353, debug=False, threaded=True)
