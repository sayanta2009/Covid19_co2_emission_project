from src.Models.model_utils import *
from keras.models import load_model
from matplotlib.figure import Figure
from matplotlib import rcParams

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../../')


class Prediction:

    def __init__(self, country, user_feature_selection):
        self.country = country
        self.feature_reduction_percentage = user_feature_selection
        self.output_graph_file = None
        self.data = pd.read_excel(MAIN_FILE)
        self.prediction_years = list(range(2019, 2031))
        self.prediction_df_without_covid19 = self.data
        self.prediction_df_with_covid19 = self.data
        self.countries = sorted(set(self.data['Country']))
        self.prediction_input_years = []
        self.model = None
        self.model_name = ''
        self.prediction_year = np.zeros((NUMBER_OF_COUNTRIES, TOTAL_NUMBER_OF_FEATURES + 3), dtype=object)
        self.country_codes = []
        self.prediction_cases = PREDICTION_CASES
        self.predict_new = True
        self.prediction_without_covid_case = False
        self.analysis_plot = False

    def create_visual_graph(self):
        """
        creates a plot and outputs path of the plot that's returned to the user interface
        :rtype: string
        """
        if self.predict_new and self.prediction_without_covid_case:
            self.predict_co2_emission_future()
            self.save_prediction_df()
        else:
            self.restore_prediction_df()
            if not self.analysis_plot:
                self.predict_co2_emission_future()
                self.save_prediction_df()

        self.do_plot()
        self.output_graph_file = OUTPUT_GRAPH_PATH
        return self.output_graph_file

    def predict_co2_emission_future(self):
        self.initialize_prediction_df()
        if self.predict_new and self.prediction_without_covid_case:
            for _, scenario in enumerate(self.prediction_cases):
                self.execute_prediction(scenario)
        else:
            self.execute_prediction('with covid')

    def execute_prediction(self, scenario):
        """
        :param scenario: string
        Considers two scenarios with and without Covid
        """
        for predict_year in self.prediction_years:
            self.prediction_year[:, 0] = np.ones(NUMBER_OF_COUNTRIES) * predict_year
            self.prediction_year[:, 1] = self.countries
            self.prediction_year[:, 18] = self.country_codes
            self.prediction_input_years = list(range(predict_year - 3, predict_year))
            if scenario == 'without covid':
                predict_df = self.prediction_df_without_covid19[
                    self.prediction_df_without_covid19['Year'].isin(self.prediction_input_years)]
            else:
                predict_df = self.prediction_df_with_covid19[
                    self.prediction_df_with_covid19['Year'].isin(self.prediction_input_years)]

            for index, feature_name in enumerate(predict_df.columns):
                if feature_name not in ('Year', 'Country', 'Country_Code'):
                    model_type = FEATURES_BEST_PIPELINE_MODEL[index - 2]
                    if feature_name == 'Total_CO2_Emissions':
                        self.retrieve_model('main_model')
                        util = ModelUtil(predict_df, LOOK_BACK_TIME_STEPS, feature_name=None)
                        util.prepare_data_main_model()
                        model_input, _ = util.create_time_series_dataset_main_model(predict=True)
                        model_input = model_input.reshape(model_input.shape[0], model_input.shape[1], -1)
                        prediction = self.model.predict(model_input)
                    else:
                        self.retrieve_model(feature_name)
                        util = ModelUtil(predict_df, LOOK_BACK_TIME_STEPS, feature_name=feature_name)
                        util.prepare_feature(self.prediction_input_years)
                        model_input, _ = util.create_time_series_dataset(predict=True)
                        model_input = model_input[:, :, 1:15]
                        if model_type == 'cnn_2':
                            model_input = model_input.reshape(model_input.shape[0], model_input.shape[1],
                                                              model_input.shape[2], 1)
                        prediction = self.model.predict(model_input)

                    self.prediction_year[:, index] = prediction
                    print('Generated prediction successfully for ', feature_name, ' and year ', predict_year, ' and ',
                          scenario)

            if scenario == 'with covid' and predict_year == 2020:
                self.apply_covid_changes()

            year_predict_df = pd.DataFrame(data=self.prediction_year, columns=predict_df.columns)
            if scenario == 'without covid':
                self.prediction_df_without_covid19 = self.prediction_df_without_covid19.append(year_predict_df)
            else:
                self.prediction_df_with_covid19 = self.prediction_df_with_covid19.append(year_predict_df)

    def retrieve_model(self, feature_name):
        self.model_name = 'src/Models/model_checkpoints/best_models/best_model_' + feature_name.strip()
        self.model = load_model(self.model_name, custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                                 'r2_keras': r2_keras})
        print(feature_name + ' retrieve success!!')

    def initialize_prediction_df(self):
        """
        returns list of country codes.

        :rtype: list country_codes
        """
        encoder = LabelEncoder()
        if self.predict_new and self.prediction_without_covid_case:
            self.prediction_df_without_covid19['Country_Code'] = encoder.fit_transform(
                self.prediction_df_without_covid19['Country'])
            self.prediction_df_without_covid19 = self.prediction_df_without_covid19. \
                sort_values(['Year', 'Country_Code'], ascending=[True, True])
            self.prediction_df_without_covid19.reset_index(drop=True, inplace=True)
            self.prediction_df_with_covid19 = self.prediction_df_without_covid19
        else:
            self.prediction_df_with_covid19['Country_Code'] = encoder.fit_transform(
                self.prediction_df_with_covid19['Country'])
            self.prediction_df_with_covid19 = self.prediction_df_with_covid19.sort_values(
                ['Year', 'Country_Code'],
                ascending=[True, True])
            self.prediction_df_with_covid19.reset_index(drop=True, inplace=True)
        self.country_codes = sorted(set(self.prediction_df_without_covid19['Country_Code']))

    def do_plot(self):
        """
        Creates and saves a graph of the prediction according to a Country.
        """
        years = sorted(set(self.prediction_df_without_covid19['Year']))
        predict_without_covid_country = self.prediction_df_without_covid19[
            self.prediction_df_without_covid19['Country'].isin([self.country])].sort_values(['Year'],
                                                                                            ascending=[True])
        predict_with_covid_country = self.prediction_df_with_covid19[
            self.prediction_df_with_covid19['Country'].isin([self.country])].sort_values(['Year'],
                                                                                         ascending=[True])
        # ------------------------------------------------------------------------------------------------------
        pa = \
            predict_without_covid_country.loc[predict_without_covid_country['Year'] == 1990][
                'Total_CO2_Emissions'].values[
                0]
        x = []
        for i in range(len(years)):
            x.append(pa * 0.6)
        # ------------------------------------------------------------------------------------------------------
        fig = Figure()
        ax = fig.subplots()
        ax.grid(True, alpha=0.3)
        # plot_title = 'Total CO2 Emissions predicted from 2019-2030 for ' + self.country
        plot_title = 'Total ' + '$CO_2$' + ' Emissions predicted from 2019-2030 for ' + self.country
        label_country_without_covid = 'Total CO2 emissions without covid'
        label_country_with_covid = 'Total CO2 emissions with Covid-19'
        # ------------------------------------------------------------------------------------------------------
        params = {'mathtext.default': 'regular'}
        rcParams.update(params)
        rcParams['font.size'] = 7
        rcParams['lines.markersize'] = 4
        rcParams['figure.figsize'] = [7, 4]
        rcParams['figure.dpi'] = 150
        rcParams['font.family'] = 'Verdana'
        rcParams["font.weight"] = "normal"
        font = {'family': 'Verdana',
                'color': 'xkcd:darkgreen',
                'weight': 'normal',
                'size': 9,
                }
        colors = rcParams['axes.prop_cycle'].by_key()['color']
        l1, = ax.plot(years, predict_without_covid_country['Total_CO2_Emissions'], color='xkcd:dark blue green',
                      marker='o',
                      label=label_country_without_covid)
        l2, = ax.plot(years, predict_with_covid_country['Total_CO2_Emissions'], color='xkcd:neon pink', marker='.',
                      label=label_country_with_covid)
        l3, = ax.plot(years, x, color='xkcd:orchid', marker='1')
        print('without covid: ', predict_without_covid_country['Total_CO2_Emissions'].values)
        print('with covid: ', predict_with_covid_country['Total_CO2_Emissions'].values)
        ax.set_xlabel('Years', fontdict=font)
        ax.set_ylabel('Emissions (Gg)', fontdict=font)
        ax.set_title(plot_title, fontsize=12, fontweight='normal')
        ax.patch.set_facecolor('xkcd:green')
        ax.set_facecolor('xkcd:pale green')
        fig.legend((l1, l2, l3), ('Prediction without Covid19', 'Prediction with Covid19', 'Paris Agreement'),
                   bbox_to_anchor=(0.907, 0.89))
        fig.savefig(OUTPUT_GRAPH_PATH)

    def apply_covid_changes(self):
        for index, reduction in enumerate(self.feature_reduction_percentage):
            without_covid_2019_predictions = self.prediction_year[:, index + 2]
            self.prediction_year[:, index + 2] = without_covid_2019_predictions * (1 - reduction / 100)

    def save_prediction_df(self):
        self.prediction_df_without_covid19.to_pickle(PREDICT_WITHOUT_COVID_FILE, protocol=2)
        self.prediction_df_with_covid19.to_pickle(PREDICT_WITH_COVID_FILE, protocol=2)

    def restore_prediction_df(self):
        self.prediction_df_without_covid19 = pd.read_pickle(PREDICT_WITHOUT_COVID_FILE)
        if self.analysis_plot:
            self.prediction_df_with_covid19 = pd.read_pickle(PREDICT_WITH_COVID_FILE)
