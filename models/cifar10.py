#from keras.applications.inception_v3 import InceptionV3 as KerasInceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
from keras.datasets import cifar10
from keras.models import model_from_json

import os.path


import config
from .base_model import BaseModel


class CIFAR10(BaseModel):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        self.num_classes=10
        self.epochs=1
        self.img_size = (32,32)

    def getVanillaCNN(self):
        self.filename = "cifar10_model.json"
        if(os.path.isfile(self.filename)):
            print("Model Json Exists!!")
            json_file = open(self.filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            self.model= loaded_model
        else:
            print("Model Json does not exists!!")
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.img_size + (3,)))
            self.model.add(Dropout(0.2))
            self.model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            self.model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Flatten())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.num_classes, activation='softmax'))
            sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
            ## save model architecture in json
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            json_string = self.model.to_json()
            json_file = open('cifar10_model.json', 'w')
            json_file.write(json_string)
            json_file.close()
            print("Model Json saved!!")

        self.filename = "cifar10_model.h5"
        if(os.path.isfile(self.filename)):    
            # load weights into new model
            self.model.load_weights(self.filename)
            print("Loaded model from disk   ....")
        else:
            print("Model weights file does not exists!!")
            print("training Model on cifar")
            # Fit model
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            # Convert and pre-processing
            y_train = np_utils.to_categorical(y_train, self.num_classes)
            y_test = np_utils.to_categorical(y_test, self.num_classes)
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train  /= 255
            x_test /= 255           
            self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_test,y_test),shuffle=True)
            base_model = self.model
            #self.make_net_layers_non_trainable(base_model)
            x = base_model.output
            predictions = Dense(self.num_classes, activation='softmax')(x)
            self.model = Model(input=base_model.input, output=predictions)
            self.model.save("cifar_model.h5")

    def _create(self):
        self.getVanillaCNN()
        base_model = self.model
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='elu', name='fc1')(x)
        x = Dropout(0.6)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.6)(x)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)

        self.model = Model(input=base_model.input, output=predictions)


def inst_class(*args, **kwargs):
    return CIFAR10(*args, **kwargs)
