from src.Models.model_utils import *
from src.Models.constants import *
import os
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import keras
from keras.layers import Dense, Flatten
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../../')
feature_name = 'main_model'
data_df = pd.read_excel(MAIN_FILE)
x_train, x_test, y_train, y_test, true_data = prepare_input_for_tuning(data_df, feature_name)
y_train_com = y_train
y_train = y_train[:, :, 0]


def build_mlp_model(optimizer=ADAM_OPTIMIZER, activation='relu'):
    model = keras.Sequential()
    model.add(Flatten(input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(128, activation=activation, name="Hidden_Layer_1"))
    model.add(Dense(64, activation=activation, name="Hidden_Layer_2"))
    model.add(Dense(16, activation=activation, name="Hidden_Layer_3"))
    model.add(Dense(units=NUMBER_OF_COUNTRIES, activation=LINEAR_ACTIVATION, name="output"))
    model.compile(optimizer=optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
    return model


plot_evaluation = True
search = False
prediction_result = np.zeros((y_test.shape[0], NUMBER_OF_COUNTRIES+1))

optimizers = [RMS_OPTIMIZER, ADAM_OPTIMIZER]
batch_size = [5, 10]
epochs = [150, 200, 250]
activation_fn = ['relu', 'tanh']
param_grid = dict(activation=activation_fn, optimizer=optimizers, batch_size=batch_size, epochs=epochs)

if plot_evaluation:
    tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_main_model',
                             custom_objects={'root_mean_squared_error': root_mean_squared_error, 'r2_keras': r2_keras})
    y_prediction = tuned_model.predict(x_test)
    test_years = y_test[:, 1][:, 1]
    prediction_result[:, 0] = test_years
    for year in range(len(test_years)):
        prediction_result[year, 1:] = y_prediction[year]

    pred_reshaped = np.zeros((7, NUMBER_OF_COUNTRIES, 2))

    for j in range(prediction_result.shape[0]):  # iterating over the years
        temp_year = prediction_result[j, 0]
        pred_reshaped[j, :, 1] = np.ones(NUMBER_OF_COUNTRIES) * temp_year
        pred_reshaped[j, :, 0] = prediction_result[j, 1:]

    y_line = np.vstack((y_train_com, y_test))
    fig, axs = plt.subplots(5, 3, figsize=(26, 37))
    for m in range(y_line.shape[1]):
        zzz = y_line[:, m, :]
        zzz = zzz[zzz[:, 1].argsort()]
        xxx = zzz[:, 1]
        yyy = zzz[:, 0]
        axs[m % 5, m // 5].plot(xxx, yyy, label='observed data')
        zz = pred_reshaped[:, m, :]
        zz = zz[zz[:, 1].argsort()]
        xx = zz[:, 1]
        yy = zz[:, 0]
        axs[m % 5, m // 5].scatter(xx, yy, label='predicted test data', c='r')
        axs[m % 5, m // 5].set_xlabel('year', fontsize=18)
        axs[m % 5, m // 5].tick_params(axis='x', labelsize=18)
        axs[m % 5, m // 5].tick_params(axis='y', labelsize=18)
        axs[m % 5, m // 5].set_ylabel('Total CO2 emission in Gg', fontsize=18)
        axs[m % 5, m // 5].legend(loc='best')
        axs[m % 5, m // 5].set_title(COUNTRIES[m], fontsize=20)
        axs[m % 5, m // 5].ticklabel_format(axis='y', style="sci", scilimits=(0,0))
        axs[m % 5, m // 5].grid()
    plt.savefig('src/Models/model_checkpoints/train_history/evaluation_plot.pdf', bbox_inches="tight")
    plt.show()
else:
    if search:
        reg_model = KerasRegressor(build_fn=build_mlp_model, verbose=1)
        grid = GridSearchCV(estimator=reg_model, param_grid=param_grid, cv=5, verbose=1)
        grid_result = grid.fit(x_train, y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_batch_size = grid_result.best_params_['batch_size']
        best_epochs = grid_result.best_params_['epochs']
        best_optimizer = grid_result.best_params_['optimizer']
        best_activation = grid_result.best_params_['activation']
        best_model = build_mlp_model(best_optimizer, best_activation)
        train_history = best_model.fit(x_train, y_train, epochs=best_epochs, batch_size=best_batch_size,
                                       validation_split=VALIDATION_SPLIT_SIZE, verbose=1)
        np.save('src/Models/model_checkpoints/train_history/main_model.npy', train_history.history)
        np.save('src/Models/model_checkpoints/train_history/main_model_best_params.npy', grid_result.best_params_)
    else:
        best_parameters = np.load('src/Models/model_checkpoints/train_history/main_model_best_params.npy',
                                  allow_pickle=True).item()
        best_optimizer = best_parameters['optimizer']
        best_activation = best_parameters['activation']
        best_epochs = best_parameters['epochs']
        best_batch_size = best_parameters['batch_size']
        best_model = build_mlp_model(best_optimizer, best_activation)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
        mc = ModelCheckpoint('src/Models/model_checkpoints/best_models/best_model_main_model',
                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [es, mc]
        best_model.fit(x_train, y_train, epochs=best_epochs, batch_size=best_batch_size,
                       validation_split=VALIDATION_SPLIT_SIZE, callbacks=callbacks_list, verbose=1)
        best_model.save('src/Models/model_checkpoints/best_models/best_model_main_model')
        y_test = y_test[:, :, 0]
        best_model.evaluate(x_test, y_test)
