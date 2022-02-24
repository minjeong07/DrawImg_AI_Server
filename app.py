
from flask import Flask, current_app, jsonify, send_file, request
app = Flask(__name__, static_url_path='')

import io
import numpy as np


from gau import evaluate, to_image
from nst import nst_apply

import tensorflow_hub as hub

from pipo import views



hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

app.register_blueprint(views)


@app.route('/generate', methods = ['POST'])
def generate():
    labelmap = np.asarray(request.json['data'])

    image = evaluate(labelmap)
    
    return {'url':image}

@app.route('/')
def index():
    return current_app.send_static_file('index.html')

@app.route('/style', methods = ['POST'])
def style():
    url = request.json['url']
    trans_style = nst_apply(url, hub_module)
    return {'result':trans_style}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
