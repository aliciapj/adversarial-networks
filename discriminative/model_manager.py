# https://github.com/gregchu

import os
import logging
import argparse
import requests

import numpy as np

from PIL import Image
from io import BytesIO

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import SGD

from settings import *

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

classes = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup',
           'Vegetable/Fruit', 'Non food']
class_to_ix = dict(zip(classes, range(len(classes))))
ix_to_class = dict(zip(range(len(classes)), classes))


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """

    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(source_path):

    training_path = os.path.join(source_path, 'training/')
    validation_path = os.path.join(source_path, 'validation/')

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

    train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
    )

    nb_classes = train_generator.num_classes
    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)


    # transfer learning
    setup_to_transfer_learn(model, base_model)
    nb_train_samples = train_generator.samples
    nb_val_samples = validation_generator.samples
    early_tl = EarlyStopping(monitor="val_loss", patience=3)
    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=NB_EPOCHS_TRANSFERLEARNING,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto',
        callbacks=[early_tl])

    # fine-tuning
    setup_to_finetune(model)
    early_ft = EarlyStopping(monitor="val_loss", patience=3)
    history_ft = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=NB_EPOCHS_FINETUNE,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto',
        callbacks=[early_ft])

    # save model
    model.save(OUTPUT_MODEL_FILE)

    # Evaluate model
    evaluation_path = os.path.join(source_path, 'evaluation/')
    evaluation_generator = test_datagen.flow_from_directory(
        evaluation_path,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
    )
    nb_eval_samples = evaluation_generator.samples
    score = model.evaluate_generator(evaluation_generator, nb_eval_samples / BATCH_SIZE)
    logger.info("\tEvaluation data. Loss: ", score[0], "Accuracy: ", score[1])


def predict(img_str):

    target_size = (229, 229)  # fixed size for InceptionV3 architecture

    img = None
    if img_str.startswith('http://') or img_str.startswith('https://'):
        response = requests.get(img_str)
        img = Image.open(BytesIO(response.content))
    elif os.path.isfile(img_str):
        img = Image.open(img_str)
    if img is None:
        return None
    if img.size != target_size:
        img = img.resize(target_size)

    model = load_model(OUTPUT_MODEL_FILE)
    class_idx_map = {'11': 3, '10': 2, '1': 1, '0': 0, '3': 5, '2': 4, '5': 7, '4': 6, '7': 9, '6': 8, '9': 11, '8': 10}
    idx_class_map = {_v: int(_k) for _k, _v in class_idx_map.iteritems()}

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    probs = preds[0]
    pred_idx = np.argmax(probs)
    pred_class = idx_class_map[pred_idx]
    pred_class_name = ix_to_class[pred_class]
    logger.info("\tClass: '{}'. Prob: {:.5f}".format(pred_class_name, probs[pred_idx]))


def get_args():
    """
        This method parses and return arguments passed in
    :return:
    """
    parser = argparse.ArgumentParser(description='Food12 model manager')
    # Add arguments
    parser.add_argument(
        '-t', '--train', type=str, help='Path with directory containing Food Image dataset', required=False)
    parser.add_argument(
        '-p', '--predict', type=str, help='Path or URL of the image to predict', required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    args_dict = dict()
    if args.train is not None:
        args_dict["train"] = args.train
    if args.predict is not None:
        args_dict["predict"] = args.predict
    return args_dict


def run():
    args_dict = get_args()
    if "train" in args_dict:
        train(source_path=args_dict["train"])
    if "predict" in args_dict:
        predict(img_str=args_dict["predict"])


if __name__ == '__main__':
    run()
