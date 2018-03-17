import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import to_categorical

import config
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Path to image', default=None, type=str)
    parser.add_argument('--accuracy', action='store_true', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_true')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture',
                        choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
                                 config.MODEL_VGG16, config.MODEL_CIFAR10,config.MODEL_CATSDOGS])
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=500, type=int, help='How many files to predict on at once')
    args = parser.parse_args()
    return args


def get_files(path):
    print("$$$",path)
    if os.path.isdir(path):
        files = glob.glob(path + '/*/*.jpg')
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files


def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        x = model_module.load_img(i)
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])

        inputs.append(x)

    return y_true, inputs


def evaluate(path):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data_dir = path

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(150, 150),
        color_mode="rgb",
        shuffle = "false",
        class_mode='categorical',
        batch_size=1)

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    print(model.metrics_names)
    loss,acc= model.evaluate_generator(test_generator,steps=10)

    print('Test loss:', loss)
    print('Test accuracy:', acc)




if __name__ == '__main__':
    tic = time.clock()

    args = parse_args()
    print('=' * 50)
    print('Called with args:')
    print(args)

    if args.data_dir:
        config.data_dir = args.data_dir
        config.set_paths()
    if args.model:
        config.model = args.model

    util.set_img_format()
    model_module = util.get_model_class_instance()
    #model = model_module.load_pretrained_model()

    classes_in_keras_format = util.get_classes_in_keras_format()
    if(config.model=='cifar10'):
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
        print('Loss: %.3f' % scores[0]) 
        print('Accuracy: %.3f' % scores[1])
    else:
        evaluate(args.path)

    if args.execution_time:
        toc = time.clock()
        print('Time: %s' % (toc - tic))


