from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        """
        Initializes a new instance of the class.
        """
        pass

    def main(self):
        """
        Executes the main function.

        This function initializes the ConfigurationManager to access the configuration settings,
        retrieves the evaluation configuration using the `get_evaluation_config()` method,
        creates an instance of the `Evaluation` class using the retrieved configuration,
        performs the evaluation by calling the `evaluation()` method,
        and saves the score by calling the `save_score()` method.

        Parameters:
        - None

        Returns:
        - None
        """
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
            