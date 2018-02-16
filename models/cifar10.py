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


import config
from .base_model import BaseModel


class CIFAR10(BaseModel):
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 1024
    freeze_layers_number = 6
    num_classes=10
    batch_size = 32 
    epochs =5
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
    def __init__(self, *args, **kwargs):
        #super(InceptionV3, self).__init__(*args, **kwargs)

        if not self.freeze_layers_number:
            self.freeze_layers_number = 5

        self.img_size = (32, 32)

    def _create(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Convert and pre-processing
        num_classes=10
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train  /= 255
        x_test /= 255           
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
        model.add(Dropout(0.2))

        model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
        model.add(Dropout(0.2))

        model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
        model.add(Dropout(0.2))

        model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)

# Train model
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit model
        history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_test,y_test),shuffle=True)
        print(history)
        base_model = model
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        predictions = Dense(len(config.classes), activation='softmax')(x)
        self.model = Model(input=base_model.input, output=predictions)

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return self.preprocess_input(x)[0]

    @staticmethod
    def apply_mean(image_data_generator):
        pass

    def _fine_tuning(self):
        self.freeze_top_layers()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'])

        self.model.fit_generator(
            self.get_train_datagen(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=self.preprocess_input),
            samples_per_epoch=config.nb_train_samples,
            nb_epoch=self.nb_epoch,
            validation_data=self.get_validation_datagen(preprocessing_function=self.preprocess_input),
            nb_val_samples=config.nb_validation_samples,
            callbacks=self.get_callbacks(config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience),
            class_weight=self.class_weight)

        self.model.save(config.get_model_path())


def inst_class(*args, **kwargs):
    return CIFAR10(*args, **kwargs)
