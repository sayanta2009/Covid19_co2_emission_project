import numpy as np
from src.Models.constants import *
from keras import backend as k
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import rcParams


def r2_keras(y_true, y_prediction):
    """
    This method calculates the r2-score metric given true and predicted data
    """
    ss_res = k.sum(k.square(y_true - y_prediction))
    ss_tot = k.sum(k.square(y_true - k.mean(y_true)))
    return 1 - ss_res / (ss_tot + k.epsilon())


def root_mean_squared_error(y_true, y_prediction):
    """
    This method returns the rmse error given the true and predicted data
    """
    return k.sqrt(k.mean(k.square(y_prediction - y_true)))


def prepare_input_for_tuning(data, feature_name):
    """
    This method formats data for our models that we use for performing grid-search. 'data' is the input dataframe and
    'feature_name' signifies the feature for which we are making the data.
    This method return the training and test data
    """
    x_train, x_test, y_train, y_test = None, None, None, None
    true_data = np.zeros((NUMBER_OF_COUNTRIES, NUMBER_OF_YEARS))
    le = LabelEncoder()
    data['Country_Code'] = le.fit_transform(data['Country'])

    if feature_name == 'main_model':
        data = data.sort_values(['Year', 'Country_Code'], ascending=[True, True])
        main_util = ModelUtil(data, NUMBER_OF_YEARS, feature_name=None)
        main_util.prepare_data_main_model()
        main_data = main_util.main_model_by_country
        for country in range(NUMBER_OF_COUNTRIES):
            for year in range(main_data.shape[0]):
                true_data[country][year] = main_data[year][country, 15]

        x_main_model, y_main_model = main_util.create_time_series_dataset_main_model()
        x_main_model = x_main_model.reshape(x_main_model.shape[0], x_main_model.shape[1], -1)
        x_train, x_test, y_train, y_test = train_test_split(x_main_model, y_main_model, shuffle=True,
                                                            random_state=42,
                                                            test_size=TEST_SPLIT_SIZE)
    else:
        # data = data.sort_values(['Country_Code', 'Year'], ascending=[True, True])
        for features in data.columns:
            if features == feature_name:
                util = ModelUtil(data, NUMBER_OF_YEARS, feature_name=feature_name)
                util.prepare_feature(list(set(data['Year'])))
                x, y = util.create_time_series_dataset()
                x_train, x_test, y_train, y_test = \
                    train_test_split(x, y, shuffle=True, random_state=42, test_size=TEST_SPLIT_SIZE)
                break

    return x_train, x_test, y_train, y_test, true_data


def plot_model_history(history, ax=None, metric='loss', ep_start=1, ep_stop=None, monitor='val_loss', mode='min',
                       plt_title=None):
    """
    This method is used to plot training progress with epochs.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if ep_stop is None:
        ep_stop = len(history[metric])
    if plt_title is None:
        plt_title = metric[0].swapcase() + metric[1:] + ' During Training'

    ax.plot(np.arange(ep_start, ep_stop + 1, dtype='int'), history[metric][ep_start - 1:ep_stop])
    ax.plot(np.arange(ep_start, ep_stop + 1, dtype='int'), history['val_' + metric][ep_start - 1:ep_stop])
    # ax.set(title=plt_title)
    # ax.set(ylabel=metric[0].swapcase() + metric[1:])
    # ax.set(xlabel='Epoch')
    font = {'family': 'Verdana',
            'color': 'black',
            'weight': 'normal',
            'size': 20,
            }
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch', fontdict=font)
    ax.set_ylabel(metric[0].swapcase() + metric[1:], fontdict=font)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    ax.set_title(plt_title)
    ax.legend(['execute', 'val'], loc='upper right')


def write_stats_to_excel(scores, file, sheet_name, train):
    """
    :param scores: r2 scores and RMSE errors for execute or evaluation
    :param file: name of the file to write
    :param sheet_name: worksheet of the excel file to write
    :param train: True for execute, False for evaluation
    """
    file_exist = False
    writer = pd.ExcelWriter(file, engine='openpyxl')
    my_file = Path(file)
    # to check if a file with this name exists
    if my_file.is_file():
        book = load_workbook(file)
        writer.book = book
        file_exist = True
    if train:
        df = pd.DataFrame(scores, index=['Training_r2_score', 'Validation_r2_score'])
        if file_exist:
            for sheet in writer.book.worksheets:
                if sheet.title == sheet_name:
                    del writer.book[sheet.title]
                    break
        df.to_excel(writer, index=True, sheet_name=sheet_name)
    else:
        df = pd.DataFrame(scores, index=['Test_RMSE', 'Test_r2_score'])
        if file_exist:
            for sheet in writer.book.worksheets:
                if sheet.title == sheet_name:
                    del writer.book[sheet.title]
                    break
        df.to_excel(writer, index=True, sheet_name=sheet_name)

    writer.save()
    writer.close()


class ModelUtil:

    def __init__(self, data_df, number_of_years, feature_name):
        self.data_df = data_df
        self.feature_name = feature_name
        self.number_of_years = number_of_years
        self.feature_by_country = np.zeros((number_of_years, NUMBER_OF_COUNTRIES + 1))
        self.country_codes = set(self.data_df['Country_Code'].to_numpy())
        self.main_model_by_country = np.zeros((number_of_years, NUMBER_OF_COUNTRIES, TOTAL_NUMBER_OF_FEATURES + 1))

    def prepare_feature(self, years):
        """
        This method takes a sorted numpy array of a particular feature and converts it a matrix where each row is a year
        and each column is a country
        :return: a (29*14) matrix
        """
        self.feature_by_country[:, 0] = years
        for code in self.country_codes:
            i = 0
            for _, row in self.data_df.iterrows():
                if row['Country_Code'] == code:
                    self.feature_by_country[i][code + 1] = row[self.feature_name]
                    i += 1
                if i == self.number_of_years:
                    break

    def prepare_data_main_model(self):
        """
        This method takes a sorted numpy array of main model and converts it a 3D matrix where each row is a year
        and each column is a 14*17 matrix
        :return: a (29*14*17) matrix
        """
        years = self.data_df['Year']
        self.data_df = self.data_df.drop(['Year', 'Country', 'Country_Code'], axis=1)
        self.data_df = pd.concat([self.data_df, years], axis=1)
        data = self.data_df.values
        for i in range(self.number_of_years):
            self.main_model_by_country[i] = data[i * NUMBER_OF_COUNTRIES:(i + 1) * NUMBER_OF_COUNTRIES]

    def create_time_series_dataset(self, predict=False):
        """
        :return: x, y which we can feed to our models
        """
        data_x, data_y = [], []
        if not predict:
            for i in range(self.feature_by_country.shape[0] - LOOK_BACK_TIME_STEPS):
                a = self.feature_by_country[i:(i + LOOK_BACK_TIME_STEPS)]
                data_x.append(a)
                data_y.append(self.feature_by_country[i + LOOK_BACK_TIME_STEPS])
        else:
            for i in range(self.feature_by_country.shape[0] - LOOK_BACK_TIME_STEPS + 1):
                a = self.feature_by_country[i:(i + LOOK_BACK_TIME_STEPS)]
                data_x.append(a)

        return np.array(data_x), np.array(data_y)

    def create_time_series_dataset_main_model(self, predict=False):
        """
        :return: x, y which we can feed to our models
        """
        data_x, data_y = [], []
        if not predict:
            for i in range(self.main_model_by_country.shape[0] - LOOK_BACK_TIME_STEPS):
                a = (self.main_model_by_country[i:(i + LOOK_BACK_TIME_STEPS)])[:, :, 0: TOTAL_NUMBER_OF_FEATURES]
                data_x.append(a)
                co2_emission = (self.main_model_by_country[i + LOOK_BACK_TIME_STEPS])[:, TOTAL_NUMBER_OF_FEATURES - 1:
                                                                                         TOTAL_NUMBER_OF_FEATURES + 1]
                data_y.append(co2_emission)
        else:
            for i in range(self.main_model_by_country.shape[0] - LOOK_BACK_TIME_STEPS + 1):
                a = (self.main_model_by_country[i:(i + LOOK_BACK_TIME_STEPS)])[:, :, 0: TOTAL_NUMBER_OF_FEATURES]
                data_x.append(a)

        return np.array(data_x), np.array(data_y)
