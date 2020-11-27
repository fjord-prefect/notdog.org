import tensorflow as tf
import base64
from flask import Flask, jsonify, make_response, request, render_template, url_for
import cv2
import io, os, re
import numpy as np
from flask_cors import CORS
from yolov3.utils import *
from yolov3.configs import *

yolo = Create_Yolov3(input_size=416, CLASSES=TRAIN_CLASSES)
yolo.load_weights("checkpoints/yolov3_custom") # use custom weights

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict():
    message = request.get_json(force=True)
    image = message['image']
    extension = message['type']
    
    base64_image, bboxes = detect_image(yolo, image, input_size=416, show=False, CLASSES=TRAIN_CLASSES, score_threshold=0.3, iou_threshold=0.45)

    if len(bboxes)==0:
        label = 'NOT DOG'
    elif len(bboxes)==1:
        label = 'DOG'
    elif len(bboxes)>1:
        label = 'YES DOGS'

    response = {
        'prediction': 'data:image/{};base64,'.format(extension) + base64_image,
        'label': label
    }
    
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')
