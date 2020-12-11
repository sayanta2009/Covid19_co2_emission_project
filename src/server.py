import zipfile
from flask import Flask, request
from flask_cors import CORS
from src.Models.prediction import *
from flask import send_file
import os
from io import BytesIO
import glob
from src.Models.constants import *


app = Flask(__name__)

CORS(app)


@app.route('/')
@app.route('/predictor', methods=['POST'])
def getPredictionGraph():
    """

    @return: prediction graph for a particular Country as response to the http 'POST' call
    @rtype: response json
    """
    graphPath = OUTPUT_GRAPH_PATH
    if os.path.exists(graphPath):
        os.remove(graphPath)

    data = request.json

    country = data.get('country')
    user_selected_features = [
        data.get('Chemical_Industry'),
        data.get('Cropland'),
        data.get('Domestic_Aviation'),
        data.get('Forestland'),
        data.get('Grassland'),
        data.get('Harvested_Wood_Products'),
        data.get('International_Aviation'),
        data.get('International_Navigation'),
        data.get('Manufacturing_Industries'),
        data.get('Metal_Industry'),
        data.get('Mineral_Industry'),
        data.get('Petroleum_Refining'),
        data.get('Public_Electricity_and_Heat_Production'),
        data.get('Railways'),
        data.get('Road_Transportation')
    ]

    prediction = Prediction(country, user_selected_features)

    prediction.create_visual_graph()

    while not os.path.exists(graphPath):
        print('waiting....')


    return send_file("../"+graphPath, mimetype='image/png')


@app.route('/marketplace_models', methods=['POST'])
def getModelsByName():
    """

    @return: .zip file in byte64 format as response to the http 'POST' call
    @rtype: response json
    """
    data = request.json
    feature = data.get('feature')
    current_path = os.getcwd()
    os.chdir('src/Models/model_checkpoints/best_models')
    path = getBestModelFolderName(feature)
    memory_file = BytesIO()
    filename = 'test_archive.zip'
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:

        for root, dirs, files in os.walk(path):
            for file in files:
                zipf.write(os.path.join(root, file))
    memory_file.seek(0)
    os.chdir(current_path)
    return send_file(memory_file, attachment_filename=filename, as_attachment=True)

@app.route('/marketplace_data', methods=['POST'])
def getDatasets():
    """

    @return: .xlsx file in byte64 format as response to the http 'POST' call
    @rtype: response json
    """
    data = request.json
    data_filename = data.get('data')
    return send_file('../Dataset/imputed_data/'+data_filename+'.xlsx')

def getBestModelFolderName(feature):
    """

    @param feature: name of the feature folder to search for
    @type feature: string
    @return: folder path if found else 'not found'
    @rtype: string
    """
    for folder in glob.glob('*'+feature):
        return folder
    return 'not found'

if __name__ == '__main__':
    app.run(port=5002)
