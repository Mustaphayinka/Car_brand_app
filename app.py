# coding=utf-8
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# Flask utils
from flask import Flask, redirect, url_for, request, render_template

#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

app.config ['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        MODEL_PATH ='model.h5'

        # Load your trained model
        model = load_model(MODEL_PATH)

        img = request.files['file']

        img.save('static/file.jpg')

        img = image.load_img('static/file.jpg', target_size = (224, 224))

        x = image.img_to_array(img)

        x=x/255

        x = np.expand_dims(x, axis = 0)
        img_data = preprocess_input(x)

        model.predict(img_data)

        prediction = np.argmax(model.predict(img_data), axis = 1)



        return render_template('result.html', prediction = prediction)



if __name__ == '__main__':
    app.run(debug = True)
