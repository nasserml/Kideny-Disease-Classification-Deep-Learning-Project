import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        """
        Initializes an instance of the class.

        Args:
            config (TrainingConfig): The configuration object for training.

        Returns:
            None
        """
        self.config = config

    
    def get_base_model(self):
        """
        Loads and returns the base model.

        Returns:
            The loaded base model.

        Parameters:
            self (object): The instance of the class.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        """
    Generates a training and validation data generator for the model.

    Returns:
        train_generator (tf.keras.preprocessing.image.DirectoryIterator): The training data generator.
        valid_generator (tf.keras.preprocessing.image.DirectoryIterator): The validation data generator.
    """

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save a Keras model to a given path.

        Args:
            path (Path): The path where the model will be saved.
            model (tf.keras.Model): The Keras model to be saved.

        Returns:
            None
        """
        model.save(path)



    
    def train(self):
        """
        Trains the model using the specified train and validation generators.
        
        Parameters:
            None
        
        Returns:
            None
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

