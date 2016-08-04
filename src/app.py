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
    '''Returns priority, severity and potential duplicates.'''

    # product = request.args.get('product')
    # component = request.args.get('component')
    # assignee = request.args.get('assignee')
    # cc = request.args.get('cc')
    # short_desc = request.args.get('short_desc')
    # desc = request.args.get('desc')

    X = pd.DataFrame([[
        request.args.get('assignee'),
        request.args.get('cc'),
        request.args.get('product'),
        'trunk', # version
        request.args.get('component'),
        request.args.get('op_sys'), # op_sys
        7, # reporter_bug_cnt
        request.args.get('desc'),
        request.args.get('short_desc'),
        '-', '-']])  # priority_final and severity_final left blank

    X.columns = ['assigned_to_init', 'cc_init', 'product_init',
                 'version_init', 'component_init', 'op_sys_init',
                 'reporter_bug_cnt', 'desc_init', 'short_desc_init',
                 'severity_final', 'priority_final']

    X_severity = create_features(X.copy(deep=True), target='severity_final')
    X_severity.pop('severity_final')

    X_priority = create_features(X.copy(deep=True), target='priority_final')
    X_priority.pop('priority_final')

    # Return formatted string
    html = '''
        <h3>Predicted priority:  {} </h3>
        <h3>Predicted severity: {} </h3>
        <h3>Potential duplicates: </h3>
        <ul>
            <li>{}</li>
            <li>{}</li>
            <li>{}</li>
        </ul>
        '''.format(
            priority_model.predict(X_priority)[0],
            severity_model.predict(X_severity)[0],
            'x', 'y', 'z')
    return str(html)


if __name__ == '__main__':

    # load models
    severity_model = pickle.load(
        open('../data/severity_final_pipeline.pkl', 'r'))
    priority_model = pickle.load(
        open('../data/priority_final_pipeline.pkl', 'r'))

    # start Flask app
    app.run(host='0.0.0.0', port=5353, debug=True, threaded=True)
