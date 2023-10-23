from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        """
        Initializes a new instance of the class.

        Parameters:
            self: The object instance.
        
        Returns:
            None
        """
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """
    Route decorator for the home page.

    Returns:
        The rendered template of the 'index.html' file.
    """
    return render_template('index.html')




@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    """
    Trains the model by executing the main script and reproducing the DVC pipeline.

    Parameters:
    None

    Returns:
    str: A message indicating that the training process has been completed successfully.
    """
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """
    Handles the "/predict" route and predicts the result based on the given image.

    Parameters:
        None

    Returns:
        A JSON object containing the predicted result.
    """
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()

    app.run(host='0.0.0.0', port=8080) #for AWS


