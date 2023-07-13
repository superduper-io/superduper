import pickle
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import pprint
import os
import time

import bson
from jinja2 import Environment, BaseLoader
from flask import Flask, request, jsonify
from flask_cors import CORS

from sddb.client import client


app = Flask(__name__)
CORS(app)


@app.route('/insert_many', methods=['POST'])
def upload():
    args = request.args
    collection = client[args['database']][args['collection']]
    file = request.files['file'].read().decode('utf-8')
    json_content = json.loads(file)
    with open('logs/gui.log', 'w') as f:
        with redirect_stdout(Unbuffered(f)):
            collection.insert_many(json_content, verbose=True)
    return jsonify(msg='ok')


def test():
    for i in range(1000):
        print(f'line {i}')
        time.sleep(1)


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


@app.route('/stream', methods=['GET'])
def stream():
    def generate():
        previous = ''
        while True:
            line = '#'.join(os.popen('tail -n 50 logs/gui.log').read().split('\n'))
            if line == previous:
                time.sleep(0.1)
                continue
            yield f'data: {line}\n\n'
            previous = line
    return app.response_class(generate(), mimetype='text/event-stream')


@app.route('/list_database_names', methods=['GET'])
def list_database_names():
    dbs = client.list_database_names()
    dbs = [db for db in dbs if db not in {'config', 'admin', 'local'}
           and not db.startswith('_')]
    return jsonify(dbs)


@app.route('/list_collection_names', methods=['GET'])
def list_collection_names():
    database = request.args['database']
    collections = client[database].list_collection_names()
    collections = [collection for collection in collections
                   if not collection.startswith('_') and '._' not in collection]
    return jsonify(collections)


def replace_id_with_object_id(filter):
    for k in filter:
        if k == '_id':
            filter[k] = bson.ObjectId(filter['_id'])
        elif isinstance(filter[k], dict):
            filter[k] = replace_id_with_object_id(filter[k])
    return filter


@app.route('/create_model', methods=['POST'])
def create_model():
    manifest = json.loads(request.args['manifest'])
    collection = request.args['collection']
    database = request.args['database']
    si = request.args['semantic_index'] == 'true'
    print(f'Is semantic index {si}')
    file_bytes = request.files['file'].read()
    model = pickle.loads(file_bytes)
    assert manifest['name'].split('.pkl')[0]
    if si:
        client[database][collection].create_semantic_index({
            'name': manifest['name'],
            'models': [{'object': model, **manifest}],
            'measure': 'css'
        })
    else:
        client[database][collection].create_model(object=model, **manifest)
    return 'ok'


@app.route('/find', methods=['POST'])
def find():
    data = request.get_json()
    collection = client[data['database']][data['collection']]
    data['filter'] = replace_id_with_object_id(data['filter'])
    print(data['filter'])
    raw_c = collection.find(data['filter'], {'_outputs': 0}, raw=True, download=True)
    html_template = collection.meta['html_template']
    raw_out = []
    it = 0
    for r in raw_c:
        raw_out.append(r)
        it += 1
        if it >= data.get('n', 10):
            break
    buffer = io.StringIO()
    pprint.pprint(raw_out, buffer)
    html_template = Environment(loader=BaseLoader).from_string(html_template)
    html = '\n'.join([
        html_template.render(r=r)
        for r in raw_out
    ])
    return jsonify({
        "printed": buffer.getvalue(),
        "html": html,
    })


if __name__ == '__main__':
    app.run(host='localhost', port=8001)
