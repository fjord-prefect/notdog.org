import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

import pickle
import base64
from flask import Flask, jsonify, make_response, request, render_template, url_for
import cv2
import io, os, re
import numpy as np
from flask_cors import CORS

model = tf.keras.models.load_model('checkpoints/model-07-0.71.hdf5')
IMG_SIZE = 250
with open('class_dict.pkl', 'rb') as handle:
    class_dict = pickle.load(handle)
LABEL_DICT = {v: k for k, v in class_dict.items()}

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict_path():
    message = request.get_json(force=True)
    image = message['image']
    extension = message['type']
    
    original_image = cv2.imdecode(np.frombuffer(base64.b64decode(image), dtype='uint8'), 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    array = cv2.resize(original_image, (IMG_SIZE,IMG_SIZE))
    model_out = model.predict(np.expand_dims(preprocess_input(array),axis=0))
    breed = re.sub('_',' ',LABEL_DICT[np.argmax(model_out)]).title()
    response = {
        'breed': breed
    }
    
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8181')

