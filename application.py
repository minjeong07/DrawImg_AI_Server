
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from gau import evaluate
from nst import nst_apply
from pipo import convert

application = Flask(__name__, static_url_path='')
CORS(application)


@application.route('/generate', methods = ['POST'])
def generate():
    labelmap = np.asarray(request.json)
    image = evaluate(labelmap)
    return {'url':image}

@application.route('/style', methods = ['POST'])
def style():
    url = request.form['url']
    trans_style = nst_apply(url)
    return {'result':trans_style}

@application.route('/pipo', methods = ['POST'])
def pipo():
    type = request.headers['Content-Type']
    if type == 'application/x-www-form-urlencoded; charset=UTF-8':
        job = 'sketch'
        url = request.form.get('url')
    elif type == 'application/json':
        job = 'pipo'
        url = request.json['url']

    result, img = convert(job, url)
    if result == 'sketch':
        return jsonify(blur = img)
    elif result == 'pipo':
        return jsonify(img=img[0], label_img=img[1])
    
    return jsonify(msg='error')


if __name__ == "__main__":
    application.debug = True
    application.run()