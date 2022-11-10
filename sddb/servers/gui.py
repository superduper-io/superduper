from contextlib import redirect_stdout
import io
import json
import pprint
import os
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

from sddb.client import client


app = Flask(__name__)
CORS(app)


@app.route('/insert', methods=['POST'])
def upload():
    file = request.files['file'].decode('utf-8')
    json_content = json.loads(file)
    with open('logs/gui.log', 'w') as f:
        with redirect_stdout(Unbuffered(f)):
            client.ebay.documents.insert_many(json_content)
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


@app.route('/find', methods=['POST'])
def main():
    data = request.get_json()
    print(data)
    result = client.ebay.documents.find_one({}, {'page': 0})
    buffer = io.StringIO()
    pprint.pprint(result, buffer)
    return jsonify({"printed": buffer.getvalue()})


if __name__ == '__main__':
    app.run(host='localhost', port=8001)
