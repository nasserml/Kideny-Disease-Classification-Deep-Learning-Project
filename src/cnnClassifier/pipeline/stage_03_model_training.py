from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger



STAGE_NAME = "Training"



class ModelTrainingPipeline:
    def __init__(self):
        """
        Initializes a new instance of the class.
        """
        pass

    def main(self):
        """
        The main function that executes the training process.

        This function initializes a `ConfigurationManager` object and retrieves the training configuration.
        It then creates a `Training` object with the retrieved configuration.

        The function proceeds to call the `get_base_model` method of the `Training` object to obtain the base model for training.

        Next, it calls the `train_valid_generator` method of the `Training` object to generate the training and validation data.

        Finally, the function calls the `train` method of the `Training` object to start the training process.

        This function does not take any parameters and does not return anything.
        """
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
