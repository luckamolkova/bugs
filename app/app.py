from __future__ import division
from flask import Flask, render_template, request
from random import random
import numpy as np
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model', methods=['GET','POST'])
def model():
    '''Returns priority, severity and potential duplicates.'''

    product = request.args.get('product')
    component = request.args.get('component')
    assignee = request.args.get('assignee')
    cc = request.args.get('cc')
    short_desc = request.args.get('short_desc')
    desc = request.args.get('desc')
    # predict
    # X = vectorizer.transform(text)
    # pred = model.predict(X)

    # Return formatted string
    html = '''
        <h3>Predicted priority:  {} </h3>
        <h3>Predicted severity: {} </h3>
        <h3>Potential duplicates: </h3>
        <ul>
            <li>{}</li>
            <li>{}</li>
            <li>{}</li>
            <li>{}</li>
        </ul>
        '''.format(product, component, assignee, cc, short_desc, desc)
    return str(html)


if __name__ == '__main__':

    # Load the model
    #  CLF, VZS, TFIDF_VZS = load_models()

    # Start Flask app
    app.run(host='0.0.0.0', port=5353, debug=True, threaded=True)
