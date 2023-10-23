import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        """
        Initializes an instance of the class.

        Parameters:
            filename (str): The name of the file to be initialized.

        Returns:
            None
        """
        self.filename =filename


    
    def predict(self):
        """
        Predicts the class of an image using a pre-trained deep learning model.

        Returns:
            A list containing a dictionary with the predicted class of the image.
            The dictionary has the following format:
            {
                "image": <predicted_class>
            }
        """
        # load model
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]