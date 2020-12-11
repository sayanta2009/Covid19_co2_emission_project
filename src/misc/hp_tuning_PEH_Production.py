from src.Models.model_utils import *
from src.Models.constants import *
import os
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import keras
from keras.layers import Dense, LSTM, Dropout
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../../')
feature_name = 'Public_Electricity_and_Heat_Production'
data_df = pd.read_excel(MAIN_FILE)
x_train, x_test, y_train, y_test, _ = prepare_input_for_tuning(data_df, feature_name)
x_train = x_train[:, :, 1:15]
y_train = y_train[:, 1:15]


# LSTM
def build_lstm_model(hidden_layer_neurons=200, dropout_rate=0.2, optimizer=ADAM_OPTIMIZER, activation='relu'):
    model = keras.Sequential()
    model.add(LSTM(units=hidden_layer_neurons, return_sequences=True,
                   input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=hidden_layer_neurons, activation=RELU_ACTIVATION))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, name="Hidden_Layer_1"))
    model.add(Dense(64, activation=activation, name="Hidden_Layer_2"))
    model.add(Dense(16, activation=activation, name="Hidden_Layer_3"))
    model.add(Dense(units=NUMBER_OF_COUNTRIES, activation=LINEAR_ACTIVATION))
    model.compile(optimizer=optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
    return model


search = False
evaluation = True

dropout_rate_set = [0.2, 0.3, 0.4]
hidden_layer_neuron_set = [150, 200, 250]
optimizers = [RMS_OPTIMIZER, ADAM_OPTIMIZER]
batch_size = [5, 10]
epochs = [150, 200, 250]
activation_fn = ['relu', 'tanh']

param_grid = dict(activation=activation_fn, dropout_rate=dropout_rate_set, hidden_layer_neurons=hidden_layer_neuron_set,
                  optimizer=optimizers,
                  batch_size=batch_size, epochs=epochs)
if not evaluation:
    if search:
        reg_model = KerasRegressor(build_fn=build_lstm_model, verbose=1)
        grid = GridSearchCV(estimator=reg_model, param_grid=param_grid, cv=5, verbose=1)
        grid_result = grid.fit(x_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_batch_size = grid_result.best_params_['batch_size']
        best_epochs = grid_result.best_params_['epochs']
        best_neurons = best_parameters['hidden_layer_neurons']
        best_optimizer = grid_result.best_params_['optimizer']
        best_activation = grid_result.best_params_['activation']
        best_dropout = best_parameters['dropout_rate']

        best_model = build_lstm_model(best_neurons, best_dropout, best_optimizer, best_activation)
        train_history = best_model.fit(x_train, y_train, epochs=best_epochs, batch_size=best_batch_size,
                                       validation_split=VALIDATION_SPLIT_SIZE, verbose=1)
        np.save('src/Models/model_checkpoints/train_history/PEH_Production.npy', train_history.history)
        np.save('src/Models/model_checkpoints/train_history/PEH_Production_best_params.npy', grid_result.best_params_)
    else:
        best_parameters = np.load('src/Models/model_checkpoints/train_history/PEH_Production_best_params.npy',
                                  allow_pickle=True).item()
        best_optimizer = best_parameters['optimizer']
        best_activation = best_parameters['activation']
        best_epochs = best_parameters['epochs']
        best_neurons = best_parameters['hidden_layer_neurons']
        best_batch_size = best_parameters['batch_size']
        best_dropout = best_parameters['dropout_rate']

        best_model = build_lstm_model(best_neurons, best_dropout, best_optimizer, best_activation)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
        mc = ModelCheckpoint('src/Models/model_checkpoints/best_models/best_model_Public_Electricity_and_Heat_Production',
                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [es, mc]
        train_history = best_model.fit(x_train, y_train, epochs=best_epochs, batch_size=best_batch_size,
                                       validation_split=VALIDATION_SPLIT_SIZE, callbacks=callbacks_list, verbose=1)
        best_model.save('src/Models/model_checkpoints/best_models/best_model_' + feature_name)

else:
    tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Public_Electricity_and_Heat_Production',
                             custom_objects={'root_mean_squared_error': root_mean_squared_error, 'r2_keras': r2_keras})
    x_test = x_test[:, :, 1:15]
    y_test = y_test[:, 1:15]
    tuned_model.evaluate(x_test, y_test)