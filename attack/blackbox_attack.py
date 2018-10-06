"""
This tutorial shows how to generate adversarial examples
using FGSM in black-box setting.
The original paper can be found at:
https://arxiv.org/abs/1602.02697
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import os

from PIL import Image

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import preprocess_input
from six.moves import xrange

import logging
import tensorflow as tf
from tensorflow.python.platform import flags
from keras.models import load_model

from cleverhans.loss import CrossEntropy
from cleverhans.model import Model
from cleverhans.utils import to_categorical
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import train, model_eval, batch_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_keras import cnn_model

from cleverhans_tutorials.tutorial_models import HeReLuNormalInitializer
from cleverhans.utils import TemporaryLogLevel

CUDA_VISIBLE_DEVICES=0,1
FLAGS = flags.FLAGS


class ModelSubstitute(Model):
    def __init__(self, scope, nb_classes, nb_filters=200, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters

    def fprop(self, x, **kwargs):
        del kwargs
        my_dense = functools.partial(
            tf.layers.dense, kernel_initializer=HeReLuNormalInitializer)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = tf.layers.flatten(x)
            y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
            y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
            logits = my_dense(y, self.nb_classes)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              aug_batch_size, rng):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = ModelSubstitute('model_s', nb_classes)
    preds_sub = model_sub.get_logits(x)
    loss_sub = CrossEntropy(model_sub, smoothing=0)

    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
            train(sess, loss_sub, x, y, X_sub,
                  to_categorical(Y_sub, nb_classes),
                  init_all=False, args=train_params, rng=rng,
                  var_list=model_sub.get_params())

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          lmbda_coef * lmbda, aug_batch_size)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
      images: array with minibatch of images
      filenames: list of filenames without path
        If number of file names in this list less than number of images in
        the minibatch then only first len(filenames) images will be saved.
      output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f)


def blackbox_attack(input_path, out_path, batch_size=128, learning_rate=0.001, data_aug=10,
                    nb_epochs_s=10, lmbda=0.1, aug_batch_size=512):
    """
    Black-box attack from arxiv.org/abs/1602.02697
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Create TF session
    sess = tf.Session()

    # data prep
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    IM_WIDTH, IM_HEIGHT = 299, 299
    BATCH_SIZE = 32

    train_generator = train_datagen.flow_from_directory(
        '{}/train'.format(input_path),
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
    )
    validation_generator = test_datagen.flow_from_directory(
        '{}/eval'.format(input_path),
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
    )
    # Get data (only one batch)
    x_train, y_train = train_generator.next()
    x_test, y_test = validation_generator.next()

    # Initialize substitute training set reserved for adversary
    X_sub = x_test
    Y_sub = np.argmax(y_test, axis=1)

    # Obtain Image parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2017, 8, 30])

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Setting up the black-box model.")
    model = load_model('/data/pycon18/src/discriminative/inceptionv3-ft120_910acc.model')
    kmodel = KerasModelWrapper(model)
    bbox_preds = kmodel.get_probs(x)

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    train_sub_out = train_sub(sess, x, y, bbox_preds, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, aug_batch_size,
                              rng)
    model_sub, preds_sub = train_sub_out

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, x_test, y_test, args=eval_params)
    accuracies['sub'] = acc

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    eval_params = {'batch_size': batch_size}
    x_adv_sub = fgsm.generate(x, **fgsm_par)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, kmodel.get_probs(x_adv_sub),
                          x_test, y_test, args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated using the substitute: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy

    adv_images = sess.run(x_adv_sub, feed_dict={x: x_test})

    original_pred = model.predict(x_test)
    attack_pred = model.predict(adv_images)

    # Save the images only if they cheat the oracle
    img_to_save = []
    for op, ap, x_img, adv_img in zip(np.argmax(original_pred, axis=1),
                                      np.argmax(attack_pred, axis=1), x_test, adv_images):
        if op != ap:
            img_to_save.append((x_img, adv_img))

    x_filenames = ['file_{}.jpg'.format(i) for i in range(len(img_to_save))]
    x_filenames_attack = ['file_{}_attack.jpg'.format(i) for i in range(len(img_to_save))]
    save_images(np.array([x[0] for x in img_to_save]), x_filenames, out_path)
    save_images(np.array([x[1] for x in img_to_save]), x_filenames_attack, out_path)

    return accuracies


def main(argv=None):
    blackbox_attack(
        input_path=FLAGS.input_path,
        out_path=FLAGS.out_path,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
        lmbda=FLAGS.lmbda, aug_batch_size=FLAGS.data_aug_batch_size
    )


if __name__ == '__main__':
    # General flags
    flags.DEFINE_string('input_path', 500, 'Input images path')
    flags.DEFINE_string('out_path', 500, 'Path for the adversarial images')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    # Flags related to substitute
    flags.DEFINE_integer('data_aug', 5, 'Nb of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 100, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')
    flags.DEFINE_integer('data_aug_batch_size', 512,
                         'Batch size for augmentation')

    tf.app.run()
