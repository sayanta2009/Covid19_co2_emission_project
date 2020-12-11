from src.Models.model_utils import *
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GRU
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.models import load_model

'''
Class to make GRU models
'''


class gru_model:

    def __init__(self, data):
        self.data = data
        self.model = None
        self.model_name = ''
        self.x = np.zeros((NUMBER_OF_YEARS - LOOK_BACK_TIME_STEPS, LOOK_BACK_TIME_STEPS, NUMBER_OF_COUNTRIES))
        self.y = np.zeros((NUMBER_OF_YEARS - LOOK_BACK_TIME_STEPS, NUMBER_OF_COUNTRIES))
        self.x_main_model = np.zeros((NUMBER_OF_YEARS - LOOK_BACK_TIME_STEPS, LOOK_BACK_TIME_STEPS, NUMBER_OF_COUNTRIES,
                                      TOTAL_NUMBER_OF_FEATURES))
        self.y_main_model = np.zeros((NUMBER_OF_YEARS - LOOK_BACK_TIME_STEPS, 1))
        self.r2_score = dict()
        self.feature_name = ''
        self.years = list(set(data['Year']))

    def make_gru_model(self, train, main_model):
        """
        This method is used to restructure our data required to feed into GRU models. 'train' is the flag to
        distinguish between training and evaluation, 'main_model' is the flag which signifies whether we are
        evaluating for the main model or the feature models
        """
        if not main_model:
            for feature_name in self.data.columns:
                if feature_name not in ('Year', 'Country', 'Country_Code', 'Energy_CO2_Emissions',
                                        'Industrial_Process_Emissions', 'Land_Use_Emissions', 'Transport_Emissions'):
                    self.feature_name = feature_name
                    util = ModelUtil(self.data, NUMBER_OF_YEARS, feature_name)
                    util.prepare_feature(self.years)  # 29*14
                    self.x, self.y = util.create_time_series_dataset()
                    x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, shuffle=True, random_state=42,
                                                                        test_size=TEST_SPLIT_SIZE)
                    self.x = self.x[:, :, 1:15]
                    self.y = self.y[:, 1:15]
                    x_train = x_train[:, :, 1:15]
                    y_train = y_train[:, 1:15]
                    self.build_gru_features_model()
                    print(self.model.summary())
                    self.model_name = 'src/Models/model_checkpoints/gru/model_gru_' + self.feature_name
                    if train:
                        self.train_gru_model(x_train, y_train)
                    else:
                        self.test_gru_model(x_test, y_test, False)
        else:
            self.feature_name = 'main_model'
            main_util = ModelUtil(self.data, NUMBER_OF_YEARS, feature_name=None)
            main_util.prepare_data_main_model()
            self.x_main_model, self.y_main_model = main_util.create_time_series_dataset_main_model()
            self.x_main_model = self.x_main_model.reshape(self.x_main_model.shape[0], self.x_main_model.shape[1], -1)
            x_train, x_test, y_train, y_test = train_test_split(self.x_main_model, self.y_main_model, shuffle=True,
                                                                random_state=42,
                                                                test_size=TEST_SPLIT_SIZE)
            y_train = y_train[:, :, 0]
            self.build_gru_main_model()
            print(self.model.summary())
            self.model_name = 'src/Models/model_checkpoints/gru/model_gru_' + self.feature_name
            if train:
                self.train_gru_model(x_train, y_train)
            else:
                self.test_gru_model(x_test, y_test, True)

        return self.r2_score

    def train_gru_model(self, x_train, y_train):
        """
        This method is used to train GRU models given training data
        """
        print('-------------Training-----------------: ', self.model_name)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
        mc = ModelCheckpoint(self.model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [es, mc]
        history = self.model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                 validation_split=VALIDATION_SPLIT_SIZE, callbacks=callbacks_list, verbose=1)
        trained_epoch = len(history.history['r2_keras'])
        self.r2_score[self.feature_name] = ((history.history['r2_keras'][trained_epoch - 1]),
                                            (history.history['val_r2_keras'])[trained_epoch - 1])
        self.save_model()

    def build_gru_features_model(self):
        """
        Builds GRU models for features
        """
        self.model = Sequential()
        self.model.add(GRU(units=HIDDEN_LAYER_NEURONS, return_sequences=True,
                           input_shape=(self.x.shape[1], self.x.shape[2]), name="gru1"))
        self.model.add(GRU(128, name="gru2"))
        self.model.add(Dense(128, activation=RELU_ACTIVATION, name="Hidden_Layer_1"))
        self.model.add(Dense(64, activation=RELU_ACTIVATION, name="Hidden_Layer_2"))
        self.model.add(Dense(16, activation=RELU_ACTIVATION, name="Hidden_Layer_3"))
        self.model.add(Dense(units=NUMBER_OF_COUNTRIES, activation=LINEAR_ACTIVATION, name="output"))
        self.model.compile(loss=root_mean_squared_error, optimizer=ADAM_OPTIMIZER, metrics=[r2_keras])

    def build_gru_main_model(self):
        """
        Builds GRU model for the main CO2 emission data.
        """
        self.model = Sequential()
        self.model.add(GRU(units=HIDDEN_LAYER_NEURONS_MAIN_MODEL, return_sequences=True,
                           input_shape=(self.x_main_model.shape[1], self.x_main_model.shape[2])))
        self.model.add(GRU(128, name="gru2"))
        self.model.add(Dense(128, activation=RELU_ACTIVATION, name="Hidden_Layer_1"))
        self.model.add(Dense(64, activation=RELU_ACTIVATION, name="Hidden_Layer_2"))
        self.model.add(Dense(16, activation=RELU_ACTIVATION, name="Hidden_Layer_3"))
        self.model.add(Dense(units=NUMBER_OF_COUNTRIES, activation=LINEAR_ACTIVATION, name="output"))
        self.model.compile(loss=root_mean_squared_error, optimizer=ADAM_OPTIMIZER, metrics=[r2_keras])
        plot_model(self.model, show_shapes=True, show_layer_names=True, dpi=128, to_file=GRU_FEATURE_MODEL_STRUCTURE)

    def save_model(self):
        """
        This method saves a keras model into disk
        """
        self.model.save(self.model_name)
        print('Saved model to disk ', self.model_name)

    def test_gru_model(self, x_test, y_test, main_model):
        """
        This method is used to evaluate the GRU models. 'main_model' is the flag which signifies whether we are
        evaluating for the main model or the feature models
        """
        print('-------Evaluating------------:', self.model_name)
        self.model = load_model(self.model_name, custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                                 'OPTIMIZER': ADAM_OPTIMIZER, 'r2_keras': r2_keras})
        if not main_model:
            x_test = x_test[:, :, 1:15]
            y_test = y_test[:, 1:15]
        else:
            y_test = y_test[:, :, 0]
        self.r2_score[self.feature_name] = self.model.evaluate(x_test, y_test)
