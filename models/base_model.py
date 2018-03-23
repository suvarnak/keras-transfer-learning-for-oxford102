import numpy as np
from numpy.random import seed
np.random.seed(1337)  # for reproducibility
from tensorflow import set_random_seed
set_random_seed(1232)

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import load_model

import numpy as np
from sklearn.externals import joblib
import config
import util
from os import path


class BaseModel(object):
    def __init__(self,
                 class_weight=None,
                 nb_epoch=1000,
                 freeze_layers_number=None):
        self.model = None
        self.class_weight = class_weight
        self.nb_epoch = nb_epoch
        self.fine_tuning_patience = 10000
        self.batch_size = 32
        self.freeze_layers_number = freeze_layers_number

    def _create(self):
        raise NotImplementedError('subclasses must override _create()')


    def _fine_tuning(self):
        self.freeze_top_layers()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])

        train_data = self.get_train_datagen(rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        callbacks = self.get_callbacks(config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience)

        if util.is_keras2():
            history = self.model.fit_generator(
                train_data,
                steps_per_epoch=config.nb_train_samples / float(self.batch_size),
                epochs=self.nb_epoch,
                validation_data=self.get_validation_datagen(),
                validation_steps=config.nb_validation_samples / float(self.batch_size),
                callbacks=callbacks,
                class_weight=self.class_weight)
        else:
            history = self.model.fit_generator(
                train_data,
                samples_per_epoch=config.nb_train_samples,
                nb_epoch=self.nb_epoch,
                validation_data=self.get_validation_datagen(),
                nb_val_samples=config.nb_validation_samples,
                callbacks=callbacks,
                class_weight=self.class_weight)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print("Fine Tuning Accuracy",acc)
        print("Fine Tuning Validation Accuracy",val_acc)
        print("Fine Tuning Loss",loss)
        print("Fine Tuning Validation Loss",val_loss)

        self.model.save(config.get_finetuned_model_path())

    def train(self):
        print("Creating model...")
        self._create()
        print("Model is created")
        print("Fine tuning...")
        self._fine_tuning()
        self.save_classes()
        print("Classes are saved")

    def load(self):
        print("loading finetuned model")
        self.load_classes()
        self._create()
        self.model.load_weights(config.get_fine_tuned_weights_path())
        return self.model

    def load_pretrained_model(self):
        print("Loading pre-trained model    ******",config.get_model_path())
        self.load_classes()
        if(path.isfile(config.get_model_path())):    
            # load weights into new model
            self.model = load_model(config.get_model_path())
            #self.model.load_weights(config.get_model_path())
            print("Loaded model from disk   ....",config.get_model_path())
        else:
            print("Model weights file does not exists!!")
        return self.model

    @staticmethod
    def save_classes():
        #joblib.dump(config.classes, config.get_classes_path())
        joblib.dump(config.finetuned_classes, config.get_finetuned_classes_path())


    def get_input_tensor(self):
        if util.get_keras_backend_name() == 'theano':
            return Input(shape=(3,) + self.img_size)
        else:
            return Input(shape=self.img_size + (3,))

    @staticmethod
    def make_net_layers_non_trainable(model):
        for layer in model.layers:
            layer.trainable = False

    def freeze_top_layers(self):
        if self.freeze_layers_number:
            print("Freezing {} layers".format(self.freeze_layers_number))
            for layer in self.model.layers[:self.freeze_layers_number]:
                layer.trainable = False
                print("freezing layer", layer.name)
            for layer in self.model.layers[self.freeze_layers_number:]:
                layer.trainable = True

    @staticmethod
    def get_callbacks(weights_path, patience=10000, monitor='val_loss'):
        early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
        model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
        return [early_stopping, model_checkpoint]

    @staticmethod
    def apply_mean(image_data_generator):
        """Subtracts the dataset mean"""
        image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

    @staticmethod
    def load_classes():
        config.finetuned_classes = joblib.load(config.get_finetuned_classes_path())

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)[0]

    def get_train_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)
        return idg.flow_from_directory(config.train_dir, target_size=self.img_size, classes=config.finetuned_classes,class_mode='categorical')
    
    def get_validation_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)
        return idg.flow_from_directory(config.validation_dir, target_size=self.img_size, classes=config.finetuned_classes,class_mode='categorical')
