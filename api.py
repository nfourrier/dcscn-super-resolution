import io
import os

from flask import abort
from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file
from flask_cors import CORS
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import DCSCN
from helper import args


api = Flask(__name__)
CORS(api)

args.flags.DEFINE_string("file", "image.jpg", "Target filename")
FLAGS = args.get()

MODEL_PATH = os.environ.get("MODEL_PATH")


def load_model(flags, model_path):
    model = DCSCN.SuperResolution(FLAGS, model_name=model_path)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    model.init_all_variables()
    model.load_model(name=model_path)
    return model


# Initialize TensorFlow session.
# not sure this will work
# tf.InteractiveSession()
sess = tf.Session()


with sess.as_default():
    with sess.graph.as_default():
        model = load_model(FLAGS, MODEL_PATH)


@api.route("/healthcheck")
def healthcheck():
    return jsonify({'Status': 'All good'}), 200


@api.route("/predict", methods=['POST'])
def predict():
    # check if the post request has the file part
    if 'image' not in request.files:
        print("No file in request.")
        return abort(403)
    image_file = request.files['image']
    img = Image.open(image_file.stream)

    data = np.array(img)
    print(data.shape)

    with sess.as_default():
        with sess.graph.as_default():
            image = model.predict_im(data)
    print(image.shape)
    # Convert array to Image
    img = PIL.Image.fromarray(image[0])
    img_io = io.BytesIO()
    img.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='prediction.png'), 200


if __name__ == '__main__':
    api.run(host='0.0.0.0')
