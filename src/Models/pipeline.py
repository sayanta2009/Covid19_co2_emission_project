import pandas as pd
from src.Models.constants import *
from sklearn.preprocessing import LabelEncoder
from src.Models.lstm_model import Lstm
from src.Models.GRU_model import gru_model
from src.Models.mlp_model import Mlp
from src.Models.CNN_model import cnn_model
from src.Models.CNN_2_model import cnn_model_2
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../../')


class Pipeline:
    """
    Class to create, execute and evaluate pipeline models for first level features and the main model
    """

    def __init__(self, data_type, model_type, train=False):
        self.data_type = data_type
        self.model_type = model_type
        self.train_condition = train
        self.data = pd.DataFrame
        self.main_model = False

    def execute(self):
        if self.data_type == 0:
            self.data = pd.read_excel(ENERGY_FILE)
        elif self.data_type == 1:
            self.data = pd.read_excel(INDUSTRY_FILE)
        elif self.data_type == 2:
            self.data = pd.read_excel(LAND_USE_FILE)
        elif self.data_type == 3:
            self.data = pd.read_excel(TRANSPORT_FILE)
        elif self.data_type == 4:
            self.data = pd.read_excel(MAIN_FILE)
            self.main_model = True
        else:
            print('Invalid Data type!!')

        LE = LabelEncoder()
        self.data['Country_Code'] = LE.fit_transform(self.data['Country'])
        # sorting our dataset with year first and then country alphabetically for faster computing
        if self.data_type == 4:
            self.data = self.data.sort_values(['Year', 'Country_Code'], ascending=[True, True])
        else:
            self.data = self.data.sort_values(['Country_Code', 'Year'], ascending=[True, True])

        sector_r2_scores = dict()
        if self.model_type == 'LSTM':
            lstm = Lstm(self.data)
            sector_r2_scores = lstm.make_lstm_model(self.train_condition, self.main_model)
        elif self.model_type == 'GRU':
            gru = gru_model(self.data)
            sector_r2_scores = gru.make_gru_model(self.train_condition, self.main_model)
        elif self.model_type == 'MLP':
            mlp = Mlp(self.data)
            sector_r2_scores = mlp.make_mlp_model(self.train_condition, self.main_model)
        elif self.model_type == 'CNN':
            cnn = cnn_model(self.data)
            sector_r2_scores = cnn.make_cnn_model(self.train_condition, self.main_model)
        elif self.model_type == 'CNN_2':
            cnn_2 = cnn_model_2(self.data)
            sector_r2_scores = cnn_2.make_cnn_model(self.train_condition, self.main_model)

        return sector_r2_scores
