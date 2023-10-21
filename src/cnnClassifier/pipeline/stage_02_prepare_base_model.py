from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        """
        Initializes a new instance of the class.
        """
        pass

    def main(self):
        """
        The main function that does the following:
        - Creates a ConfigurationManager object.
        - Retrieves the prepare_base_model_config from the ConfigurationManager object.
        - Creates a PrepareBaseModel object with the prepare_base_model_config.
        - Retrieves the base model from the PrepareBaseModel object.
        - Updates the base model.

        Parameters:
        - None

        Returns:
        - None
        """
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


    
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e