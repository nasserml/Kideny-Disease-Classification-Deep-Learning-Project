import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes a new instance of the class.

        Args:
            config (PrepareBaseModelConfig): The configuration object for the model.

        Returns:
            None
        """
        self.config = config

    
    def get_base_model(self):
        """
        Retrieves the base model for the neural network.

        Returns:
            None
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare a full model by adding a classification layer on top of the given model.

        Args:
            model (tf.keras.Model): The base model to build upon.
            classes (int): The number of classes for classification.
            freeze_all (bool): If True, freeze all layers in the model. Defaults to False.
            freeze_till (int): If provided, freeze all layers in the model except the last `freeze_till` layers. Defaults to None.
            learning_rate (float): The learning rate for the optimizer. Defaults to None.

        Returns:
            tf.keras.Model: The prepared full model with the classification layer added on top.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        """
        Updates the base model by preparing a full model with the given parameters and saving it to the specified path.

        Parameters:
            None

        Returns:
            None
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the given `model` to the specified `path`.

        Parameters:
            path (Path): The path where the model will be saved.
            model (tf.keras.Model): The model to be saved.
        """
        model.save(path)

