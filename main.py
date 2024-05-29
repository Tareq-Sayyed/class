from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the model
model_path = './model.h5'
model = tf.keras.models.load_model(model_path)

# Class names for the model predictions
class_names = ['NEW', 'OLD']

def preprocess_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_img = tf.image.resize(img_rgb, (256, 256))
    preprocessed_img = resized_img / 255.0
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    return preprocessed_img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(preprocessed_image)

    if prediction > 0.5:
        class_label = class_names[1]  # 'OLD'
    else:
        class_label = class_names[0]  # 'NEW'

    return class_label

@app.route('/test', methods=['GET'])
def test():
    return "Hola"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
