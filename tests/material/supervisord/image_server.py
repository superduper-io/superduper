from flask import Flask, make_response

app = Flask(__name__)


@app.route('/<path:path>')
def image(path):
    with open(f'img/{path}', 'rb') as f:
        img = f.read()
    response = make_response(img)
    return response


if __name__ == '__main__':
    app.run('localhost', 8002)
