from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig
                                                )


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        """
    Initializes the class with the given configuration and parameter file paths.
    
    Args:
        config_filepath (str, optional): The file path of the configuration file. Defaults to CONFIG_FILE_PATH.
        params_filepath (str, optional): The file path of the parameter file. Defaults to PARAMS_FILE_PATH.
    
    Returns:
        None
    """
   

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
        


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the data ingestion configuration.

        :return: An instance of the DataIngestionConfig class containing the configuration details.
        :rtype: DataIngestionConfig
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    


    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Returns the PrepareBaseModelConfig object with the configured parameters for preparing the base model.
        
        :return: PrepareBaseModelConfig object with the following attributes:
                 - root_dir: Path to the root directory of the base model
                 - base_model_path: Path to the base model to be prepared
                 - updated_base_model_path: Path to the updated base model after preparation
                 - params_image_size: Image size parameter for the preparation
                 - params_learning_rate: Learning rate parameter for the preparation
                 - params_include_top: Include top parameter for the preparation
                 - params_weights: Weights parameter for the preparation
                 - params_classes: Classes parameter for the preparation
        :rtype: PrepareBaseModelConfig
        """
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    



    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves the training configuration for the model.

        :return: The training configuration.
        :rtype: TrainingConfig
        """
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config
    


    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Returns the evaluation configuration for the current instance.

        :return: An instance of EvaluationConfig containing the following attributes:
                 - path_of_model: Path to the model file.
                 - training_data: Path to the training data directory.
                 - mlflow_uri: URI of the MLflow server.
                 - all_params: Dictionary of all parameters.
                 - params_image_size: Image size parameter.
                 - params_batch_size: Batch size parameter.
        """
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/kidney-ct-scan-image",
            mlflow_uri="https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config

