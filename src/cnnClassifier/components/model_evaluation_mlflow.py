import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        """
        Initializes a new instance of the class with the given configuration.

        Parameters:
            config (EvaluationConfig): The configuration object for the evaluation.

        Returns:
            None
        """
        self.config = config

    
    def _valid_generator(self):
        """
    Initializes and returns a valid data generator for the model. 
    
    This function creates a valid data generator using the `ImageDataGenerator` class 
    from the `tf.keras.preprocessing.image` module. The data generator is configured 
    with the following parameters:
    
    - `rescale`: A float value representing the rescaling factor for the input images.
    - `validation_split`: A float value representing the fraction of the data to use for validation.
    
    The data generator is then used to generate a flow of validation data from the specified 
    directory. The directory is specified using the `directory` parameter, and the subset of 
    data to use is specified using the `subset` parameter. The `target_size` parameter is used 
    to specify the size to which all images will be resized. The `batch_size` parameter 
    determines the number of samples per batch, and the `interpolation` parameter specifies 
    the interpolation method to use for image resizing.
    """

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
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


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load a Keras model from the specified path.

        Args:
            path (Path): The path to the saved model.

        Returns:
            tf.keras.Model: The loaded Keras model.
        """
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        """
        Evaluates the model and saves the score.

        Returns:
            None
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """
        Save the score to a JSON file.

        This function takes no parameters.

        Returns:
            None
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        """
        Logs into MLflow and sets the registry URI to the provided MLflow URI in the configuration. 

        Args:
            self.config.mlflow_uri (str): The MLflow URI to set as the registry URI.

        Returns:
            None

        Raises:
            None
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
