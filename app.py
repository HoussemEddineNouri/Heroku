import os
import sys
import re
import base64

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
import h5py

# Some utilites
import numpy as np
from PIL import Image
from io import BytesIO

import urllib.request
import requests

# Declare a flask app
app = Flask(__name__)

URL_json = "http://download1505.mediafire.com/fojyaxfqpwxg/6yvthsj8zb52ky1/classifier.json"
json_file = urllib.request.urlopen(URL_json)
# json_file = open('http://atia.org.tn/V0/classifier.json','r')
loaded_classifier_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_classifier_json)

#load weights into new model
url = 'http://download2266.mediafire.com/as8j9vw1hpzg/jltxkara9q2g56y/classifier.h5'
r = requests.get(url)
with open('single_prediction/classifier.h5', 'wb') as f:
    f.write(r.content)

loaded_classifier.load_weights('single_prediction/classifier.h5')
# Compiling the CNN
loaded_classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Loaded classifier Model from disk")

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image

def model_predict(img, loaded_classifier):
	
    img = img.resize((200, 200))

    # Preprocessing the image
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    test_image = preprocess_input(test_image, mode='tf')

    result = loaded_classifier.predict(test_image)
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/ar', methods=['GET'])
def index_ar():
    # Main page
    return render_template('index_ar.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json).convert('RGB')
        #print(request.files['file'])

        # Save the image to ./uploads
        #img.save("uploads/image.png")

        # Make prediction
        result = model_predict(img, loaded_classifier)

        # Process your result for human
        if result[0][0] == 1:
            prediction = 'Normal'            
        else:
            prediction = 'Covid-19'   

        pred_proba = "{:.3f}".format(np.amax(result))    # Max probability
        
        # Serialize the result, you can add additional fields
        return jsonify(result=prediction, probability=pred_proba)

    return None


if __name__ == '__main__':

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
