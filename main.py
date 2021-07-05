from flask import Flask, render_template, url_for, redirect, request, flash, send_from_directory
from werkzeug.utils import secure_filename
import flask
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os




UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.config['SECRET_KEY'] = '2008de4bbf105d61f26a763f8'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




# image preprocessing and prediction.
def predict(imagepath, imagefilename):

        # some configvalues
    # set the path to the serialized model after training
    MODEL_PATH = os.path.sep.join(["model", "soil.model"])

    # initialize the list of class label names
    CLASSES = ["LoamSoil", "SandSoil", "ClaySoil"]


    # load the input image and then clone it so we can draw on it later
    image = cv2.imread(imagepath)
    originalimage = cv2.imread(imagepath)
    output = image.copy()
    output = imutils.resize(output, width=400)

    # our model was trained on RGB ordered images but OpenCV represents
    # images in BGR order, so swap the channels, and then resize to
    # 224x224 (the input dimensions for VGG16)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # convert the image to a floating point data type and perform mean
    # subtraction
    image = image.astype("float32")
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    image -= mean

    # load the trained model from disk
    print("[INFO] loading model...")
    model = load_model(MODEL_PATH)

    # pass the image through the network to obtain our predictions
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    i = np.argmax(preds)
    label = CLASSES[i]

    # draw the prediction on the output image
    text = "{}: {:.2f}%".format(label, preds[i] * 100)
    cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 0), 2)


    # save the output image
    filename = imagefilename
    imageplacedpath = './static/uploads/' + filename

    cv2.imwrite(imageplacedpath, originalimage)
    print("written successfully")


    results = {
        "filename":filename,
        "text":text
    }

    print(results)

    return results


@app.route('/')
def upload_form():
    return render_template('main.html')


@app.route('/', methods=['POST', 'GET'])
def main():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No Image selected.', 'danger')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Analytics Done!', 'success')
        image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        prediction = predict(image,filename)
        return render_template('main.html', filename=prediction['filename'], prediction=prediction['text'])
    else:
        flash('Allowed image types are: png, jpg, jpeg', 'danger')
        return redirect(request.url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run(debug=True)
