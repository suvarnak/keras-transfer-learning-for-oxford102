#from keras.applications.inception_v3 import InceptionV3 as KerasInceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
from keras.datasets import cifar10
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import os, shutil

from os import path

import config
from .base_model import BaseModel


class CATSDOGS(BaseModel):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):
        super(CATSDOGS, self).__init__(*args, **kwargs)
        self.num_classes=2
        self.epochs=10
        self.img_size = (150,150)


    def getVanillaCNN(self):
        self.filename = "model-cats_dogs_model.json"
        img_width, img_height = 150, 150
        if(path.isfile(self.filename)):
            print("Model Json Exists!!")
            json_file = open(self.filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            model= loaded_model
        else:
            print("Model Json does not exists!!")
            # dimensions of our images.

            if K.image_data_format() == 'channels_first':
                input_shape = (3, img_width, img_height)
            else:
                input_shape = (img_width, img_height, 3)

            model = Sequential()
            model.add(Conv2D(32, (3, 3), input_shape=input_shape))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(64))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])

            from keras.utils.np_utils import to_categorical
            categorical_labels = to_categorical([0,1], num_classes=2)
            print(model)
            json_string = model.model.to_json()
            json_file = open('model-cats_dogs_model.json', 'w')
            json_file.write(json_string)
            json_file.close()
            print("Model Json saved!!")

        self.filename = "./trained/model-cats_dogs_model.h5"


        if(path.isfile(self.filename)):    
            # load weights into new model
            self.model.load_weights(self.filename)
            print("Loaded model from disk   ....")
        else:
            print("Model weights file does not exists!!")
            print("training Model on cats and dogs dataset........")
            # this is the augmentation configuration we will use for training
            train_data_dir = '../code-colt/data/cats_and_dogs//train'
            validation_data_dir = '../code-colt/data/cats_and_dogs//valid'
            nb_train_samples = 2000
            nb_validation_samples = 800
            epochs = 50
            batch_size = 16


            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            # this is the augmentation configuration we will use for testing:
            # only rescaling
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical')

            validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical')

            history = model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // batch_size)
            model.save('model-cats_dogs_model2.h5')
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            print("Training Accuracy",acc)
            print("Validation Accuracy",val_acc)
            print("Loss",loss)
            print("Validation Loss",val_loss)
            self.model.save(self.filename)
    def loadVanillaCNN(self):
        if(path.isfile(config.get_model_path())):    
            # load weights into new model
            self.model = load_model(config.get_model_path())
            print("Loaded model from disk   ....",config.get_model_path())
        else:
            print("Model weights file does not exists!!")
            self.getVanillaCNN()

    def _create(self):
        self.loadVanillaCNN()
        base_model = self.model
        self.make_net_layers_non_trainable(base_model)
        x = base_model.output
        #x = Dense(512, activation='elu', name='fc1')(x)
        #x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)
        self.model = Model(input=base_model.input, output=x)
        print(self.model.summary())



def inst_class(*args, **kwargs):
    return CATSDOGS(*args, **kwargs)
