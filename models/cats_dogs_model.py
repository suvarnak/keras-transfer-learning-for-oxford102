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
        self.filename = "catsdogs_model.json"
        if(path.isfile(self.filename)):
            print("Model Json Exists!!")
            json_file = open(self.filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            self.model= loaded_model
        else:
            print("Model Json does not exists!!")
            self.model = Sequential()

            self.model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(32, (3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(64, (3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Flatten())
            self.model.add(Dense(64))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))

            self.model.compile(loss='binary_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])

            self.model.summary()
            json_string = self.model.to_json()
            json_file = open('cats_dogs_model.json', 'w')
            json_file.write(json_string)
            json_file.close()
            #print(self.model.summary())
            print("Model Json saved!!")

        self.filename = "cats_dogs_model.h5"
        if(path.isfile(self.filename)):    
            # load weights into new model
            self.model.load_weights(self.filename)
            print("Loaded model from disk   ....")
        else:
            print("Model weights file does not exists!!")
            print("training Model on cats and dogs dataset........")
            # Fit model
            #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
            from keras.preprocessing.image import ImageDataGenerator
            from os.path import join as join_path
            import os
            abspath = os.path.dirname(os.path.abspath(__file__))
            train_dir = join_path(abspath, 'pretraining_data/cats_and_dogs_small/train/')
            validation_dir = join_path(abspath, 'pretraining_data/cats_and_dogs_small/valid/')
            
            # All images will be rescaled by 1./255
            train_datagen = ImageDataGenerator(rescale=1./255)
            test_datagen = ImageDataGenerator(rescale=1./255)

            train_generator= train_datagen.flow_from_directory(
                    # This is the target directory
                    train_dir,
                    # All images will be resized to 150x150
                    target_size=(150, 150),
                    batch_size=20,
                    # Since we use binary_crossentropy loss, we need binary labels
                    class_mode='binary')

            validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=10,
                    class_mode='binary')
            history = self.model.fit_generator(
                train_generator,
                steps_per_epoch=10,
                epochs=3,
                validation_data=validation_generator,
                validation_steps=5)            
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            print("Training Accuracy",acc)
            print("Validation Accuracy",val_acc)
            print("Loss",loss)
            print("Validation Loss",val_loss)
            self.model.save("cats_dogs_model.h5")

    def _create(self):
        self.getVanillaCNN()
        base_model = self.model
        self.make_net_layers_non_trainable(base_model)
        x = base_model.output
        x = Dense(512, activation='elu', name='fc1')(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)
        self.model = Model(input=base_model.input, output=predictions)


def inst_class(*args, **kwargs):
    return CATSDOGS(*args, **kwargs)
