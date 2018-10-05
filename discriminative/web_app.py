import os
import time
import flask
import logging
import optparse
import requests

import numpy as np

import tornado.wsgi
import tornado.httpserver

from PIL import Image
from io import BytesIO

from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

from flask import request



classes = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup',
           'Vegetable/Fruit', 'Non food']
class_to_ix = dict(zip(classes, range(len(classes))))
ix_to_class = dict(zip(range(len(classes)), classes))
class_idx_map = {'11': 3, '10': 2, '1': 1, '0': 0, '3': 5, '2': 4, '5': 7, '4': 6, '7': 9, '6': 8, '9': 11,
                 '8': 10}
idx_class_map = {_v: int(_k) for _k, _v in class_idx_map.iteritems()}

# Obtain the flask app object
app = flask.Flask(__name__)
UPLOAD_FOLDER = './static'

@app.route('/')
def index():
    print("call index")
    return flask.render_template('index.html', has_result=False)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   print("call upload file")
   if request.method == 'POST':
        if not os.path.exists(UPLOAD_FOLDER):
          os.makedirs(UPLOAD_FOLDER)

        f = request.files['file']
        filename = f.filename
        _filename = '.'.join([str(time.time()), filename])
        logging.info('Image: %s', f.filename)
        _img = BytesIO()
        f.save(_img)
        img_path = os.path.join(UPLOAD_FOLDER, _filename)
        img = Image.open(_img)
        img.save(img_path)
        result = app.clf.classify_image(img)
        return flask.render_template('index.html', has_result=True, result=result, imagesrc=img_path)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    image_url = flask.request.args.get('imageurl', '')
    logging.info('Image: %s', image_url)
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    result = app.clf.classify_image(img)
    return flask.render_template('index.html', has_result=True, result=result, imagesrc=image_url)


class ImageClassifier(object):
    def __init__(self, model_file):
        logging.info('Loading InceptionV3 model...')
        self.model = load_model(model_file)

    def classify_image(self, img):
        try:

            target_size = (229, 229)  # fixed size for InceptionV3 architecture
            if img.size != target_size:
                img = img.resize(target_size)
            start_time = time.time()
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = self.model.predict(x)
            probs = preds[0]

            top_n = 5
            top_n_idx = (-probs).argsort()[:top_n]

            predictions = []
            for _idx in top_n_idx:
                _class = idx_class_map[_idx]
                _class_name = ix_to_class[_class]
                predictions.append((_class_name, '%.5f' % probs[_idx]))

            return (True, predictions, '%.3f' % (time.time() - start_time))

        except Exception as err:
            logging.info('Classification error: %s', err)
            import sys, traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            return False, 'Something went wrong when classifying the image. Maybe try another one?'


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="Enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="Which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-m', '--model',
        help="Path of the model file",
        type='str')

    opts, args = parser.parse_args()

    if not opts.model:
        logging.error('Missing model file')
        return
    app.clf = ImageClassifier(opts.model)

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
