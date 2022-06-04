import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

import flask_cors
from flask import Flask, flash, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    response="Whatever you wish too return"
    return response
PATH_TO_MODEL="Utils/model-3x3.h5"
def get_predictions(path_to_model,path_to_image):
    reconstructed_model = keras.models.load_model(path_to_model)
    # Let's check:
    scores = reconstructed_model.predict(x_input)
    print(scores[0].shape)

    prediction = np.argmax(scores)
    print('ClassId:', prediction)

    def label_text(file):
        label_list = []
        r = pd.read_csv(file)
        for name in r['SignName']:
            label_list.append(name)
        return label_list

    labels = label_text('Utils/label_names.csv')

    print('Label:', labels[prediction])



if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True,host="0.0.0.0",use_reloader=False)

flask_cors.CORS(app, expose_headers='Authorization')