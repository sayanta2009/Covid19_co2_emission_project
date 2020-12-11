import os
from keras.models import load_model
from src.Models.model_utils import *
from keras.utils.vis_utils import plot_model
from matplotlib import rcParams


page = 2 # change for saving first (1) or second (2) png page for the report

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../../')
fig1, axs = plt.subplots(4, 2, figsize=(26, 37))
rcParams['font.size'] = 20


# ---International_Aviation---#
print('International Aviation')
int_aviation_train_history = np.load('src/Models/model_checkpoints/train_history/int_aviation.npy',
                                     allow_pickle=True).item()
int_aviation_best_params = np.load('src/Models/model_checkpoints/train_history/int_aviation_best_params.npy',
                                   allow_pickle=True).item()
print(int_aviation_best_params)

int_aviation_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_International_Aviation',
                                      custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                      'r2_keras': r2_keras})
# plot_model(int_aviation_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/int_aviation_model_structure.svg')
if page == 1:
    plot_model_history(int_aviation_train_history, plt_title='International Aviation', ax=axs[0, 0])
# ---International Navigation---#
print('International Navigation')
int_navigation_train_history = np.load('src/Models/model_checkpoints/train_history/int_navigation.npy',
                                       allow_pickle=True).item()
int_navigation_best_params = np.load('src/Models/model_checkpoints/train_history/int_navigation_best_params.npy',
                                     allow_pickle=True).item()
print(int_navigation_best_params)
int_navigation_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_International_Navigation',
                                        custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                        'r2_keras': r2_keras})
# plot_model(int_navigation_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/int_navigation_model_structure.svg')
if page == 1:
    plot_model_history(int_navigation_train_history, plt_title='International Navigation', ax=axs[1, 0])

# ---Manufacturing_Industries---#
print('Manufacturing Industries')
manu_industries_train_history = np.load('src/Models/model_checkpoints/train_history/manu_industries.npy',
                                        allow_pickle=True).item()
manu_industries_best_params = np.load('src/Models/model_checkpoints/train_history/manu_industries_best_params.npy',
                                      allow_pickle=True).item()
print(manu_industries_best_params)
manu_industries_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Manufacturing_Industries'
                                         , custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                           'r2_keras': r2_keras})
# plot_model(manu_industries_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/manu_industries_model_structure.svg')
if page == 1:
    plot_model_history(manu_industries_train_history, plt_title='Manufacturing Industries', ax=axs[2, 0])

# ---Petroleum_Refining---#
print('Petroleum Refining')
petroleum_refining_train_history = np.load('src/Models/model_checkpoints/train_history/petroleum_refining.npy',
                                           allow_pickle=True).item()
petroleum_refining_best_params = np.load(
    'src/Models/model_checkpoints/train_history/petroleum_refining_best_params.npy',
    allow_pickle=True).item()
print(petroleum_refining_best_params)
petroleum_refining_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Petroleum_Refining',
                                            custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                            'r2_keras': r2_keras})
# plot_model(petroleum_refining_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/petroleum_refining_model_structure.svg')
if page == 1:
    plot_model_history(petroleum_refining_train_history, plt_title='Petroleum Refining', ax=axs[3, 0])

# ---Public_Electricity_and_Heat_Production---#
print('Public_Electricity_and_Heat_Production')
PEH_Production_train_history = np.load('src/Models/model_checkpoints/train_history/PEH_Production.npy',
                                       allow_pickle=True).item()
PEH_Production_best_params = np.load('src/Models/model_checkpoints/train_history/PEH_Production_best_params.npy',
                                     allow_pickle=True).item()
print(PEH_Production_best_params)
PEH_Production_tuned_model = load_model(
    'src/Models/model_checkpoints/best_models/best_model_Public_Electricity_and_Heat_Production',
    custom_objects={'root_mean_squared_error': root_mean_squared_error,
                    'r2_keras': r2_keras})
# plot_model(PEH_Production_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/PEH_Production_model_structure.svg')
if page == 1:
    plot_model_history(PEH_Production_train_history, plt_title='Public_Electricity_and_Heat_Production', ax=axs[0, 1])
# ---Mineral_Industry---#
print('Mineral industry')
Mineral_Industry_train_history = np.load('src/Models/model_checkpoints/train_history/Mineral_Industry.npy',
                                         allow_pickle=True).item()
Mineral_Industry_best_params = np.load('src/Models/model_checkpoints/train_history/Mineral_Industries_best_params.npy',
                                       allow_pickle=True).item()
print(Mineral_Industry_best_params)
Mineral_Industries_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Mineral_Industry',
                                            custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                            'r2_keras': r2_keras})
# plot_model(Mineral_Industries_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/Mineral_Industries_model_structure.svg')
if page == 1:
    plot_model_history(Mineral_Industry_train_history, plt_title='Mineral industry', ax=axs[1, 1])
# Metal Industry
print('Metal Industry')
metal_indus_train_history = np.load('src/Models/model_checkpoints/train_history/metal_industry.npy',
                                    allow_pickle=True).item()
metal_industry_best_params = np.load('src/Models/model_checkpoints/train_history/metal_industry_best_params.npy',
                                     allow_pickle=True).item()
print(metal_industry_best_params)
metal_industry_tuned_model = load_model(
    'src/Models/model_checkpoints/best_models/best_model_Metal_Industry',
    custom_objects={'root_mean_squared_error': root_mean_squared_error,
                    'r2_keras': r2_keras})
# plot_model(metal_industry_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/metal_industry_model_structure.svg')
if page == 1:
    plot_model_history(metal_indus_train_history, plt_title='Metal Industry', ax=axs[2, 1])
# # ---Cropland---#
print('Cropland')
cropland_train_history = np.load('src/Models/model_checkpoints/train_history/cropland.npy',
                                 allow_pickle=True).item()
cropland_best_params = np.load('src/Models/model_checkpoints/train_history/cropland_best_params.npy',
                               allow_pickle=True).item()
print(cropland_best_params)
cropland_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Cropland',
                                  custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                  'r2_keras': r2_keras})
# plot_model(cropland_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/Cropland_model_structure.svg')
if page == 1:
    plot_model_history(cropland_train_history, plt_title='Cropland', ax=axs[3, 1])
if page == 1:
    plt.savefig('src/Models/model_checkpoints/train_history/epoch_vs_loss_1.svg')
    plt.show()

# ---Chemical Industry---#

print('Chemical Industry')
chemical_industry_train_history = np.load('src/Models/model_checkpoints/train_history/chemical_industry.npy',
                                          allow_pickle=True).item()
chemical_industry_best_params = np.load('src/Models/model_checkpoints/train_history/chemical_industry_best_params.npy',
                                        allow_pickle=True).item()
print(chemical_industry_best_params)
chemical_industry_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Chemical_Industry',
                                           custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                           'r2_keras': r2_keras})
# plot_model(chemical_industry_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/chemical_industry_model_structure.svg')
if page == 2:
    plot_model_history(chemical_industry_train_history, plt_title='Chemical Industry', ax=axs[0, 0])
# ---Forestland---#
print('Forestland')
forestland_train_history = np.load('src/Models/model_checkpoints/train_history/forestland.npy',
                                   allow_pickle=True).item()
forestland_best_params = np.load('src/Models/model_checkpoints/train_history/forestland_best_params.npy',
                                 allow_pickle=True).item()
print(forestland_best_params)
forestland_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Forestland',
                                    custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                    'r2_keras': r2_keras})
# plot_model(forestland_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/forestalnd_model_structure.svg')
if page == 2:
    plot_model_history(forestland_train_history, plt_title='Forestland', ax=axs[1, 0])  # ---Grassland---#
print('Grassland')
grassland_train_history = np.load('src/Models/model_checkpoints/train_history/grassland.npy',
                                  allow_pickle=True).item()
grassland_best_params = np.load('src/Models/model_checkpoints/train_history/grassland_best_params.npy',
                                allow_pickle=True).item()
print(grassland_best_params)
grassland_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Grassland',
                                   custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                   'r2_keras': r2_keras})
# plot_model(grassland_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/grassland_model_structure.svg')
if page == 2:
    plot_model_history(grassland_train_history, plt_title='Grassland', ax=axs[2, 0])  # ---Grassland---#
# ---Harvested Wood Products---#
print('Harvested Wood Products')
harvested_wood_products_train_history = np.load(
    'src/Models/model_checkpoints/train_history/harvested_wood_products.npy',
    allow_pickle=True).item()
harvested_wood_products_best_params = np.load(
    'src/Models/model_checkpoints/train_history/harvested_wood_products_best_params.npy',
    allow_pickle=True).item()
print(harvested_wood_products_best_params)
harvested_wood_products_tuned_model = load_model(
    'src/Models/model_checkpoints/best_models/best_model_Harvested_Wood_Products',
    custom_objects={'root_mean_squared_error': root_mean_squared_error,
                    'OPTIMIZER': ADAM_OPTIMIZER, 'r2_keras': r2_keras})
# plot_model(harvested_wood_products_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/harvested_wood_products_model_structure.svg')
if page == 2:
    plot_model_history(harvested_wood_products_train_history, plt_title='Harvested Wood Products',
                       ax=axs[3, 0])  # ---Grassland---#

# ---Domestic Aviation---#
print('Domestic Aviation')
domestic_aviation_train_history = np.load('src/Models/model_checkpoints/train_history/domestic_aviation.npy',
                                          allow_pickle=True).item()
domestic_aviation_best_params = np.load('src/Models/model_checkpoints/train_history/domestic_aviation_best_params.npy',
                                        allow_pickle=True).item()
print(domestic_aviation_best_params)
domestic_aviation_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Domestic_Aviation',
                                           custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                           'OPTIMIZER': ADAM_OPTIMIZER, 'r2_keras': r2_keras})
# plot_model(domestic_aviation_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/domestic_aviation_model_structure.svg')
if page == 2:
    plot_model_history(domestic_aviation_train_history, plt_title='Domestic Aviation', ax=axs[0, 1])  # ---Grassland---#

# ---Road Transportation---#
print('Road Transportation')
road_transportation_train_history = np.load('src/Models/model_checkpoints/train_history/road_transportation.npy',
                                            allow_pickle=True).item()
road_transportation_best_params = np.load(
    'src/Models/model_checkpoints/train_history/road_transportation_best_params.npy',
    allow_pickle=True).item()
print(road_transportation_best_params)
road_transportation_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Road_Transportation',
                                             custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                             'OPTIMIZER': ADAM_OPTIMIZER, 'r2_keras': r2_keras})
# plot_model(road_transportation_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/road_transportation_model_structure.svg')
if page == 2:
    plot_model_history(road_transportation_train_history, plt_title='Road Transportation',
                       ax=axs[1, 1])  # ---Grassland---#

# ---Railways---#
print('Railways')
railways_train_history = np.load('src/Models/model_checkpoints/train_history/railways.npy',
                                 allow_pickle=True).item()
print('history load success')
railways_best_params = np.load('src/Models/model_checkpoints/train_history/railways_best_params.npy',
                               allow_pickle=True).item()
print(railways_best_params)
railways_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_Railways',
                                  custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                  'OPTIMIZER': ADAM_OPTIMIZER, 'r2_keras': r2_keras})
# plot_model(railways_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/railways_model_structure.svg')
if page == 2:
    plot_model_history(railways_train_history, plt_title='Railways', ax=axs[2, 1])

# ---Total CO2 Emissions---#
print('Total CO2 Emissions')
main_model_train_history = np.load('src/Models/model_checkpoints/train_history/main_model.npy',
                                   allow_pickle=True).item()
main_model_best_params = np.load('src/Models/model_checkpoints/train_history/main_model_best_params.npy',
                                 allow_pickle=True).item()
main_model_tuned_model = load_model('src/Models/model_checkpoints/best_models/best_model_main_model',
                                    custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                    'OPTIMIZER': ADAM_OPTIMIZER, 'r2_keras': r2_keras})
# plot_model(main_model_tuned_model, show_shapes=True, show_layer_names=True, dpi=128,
#            to_file='src/Models/model_checkpoints/train_history/main_model_model_structure.svg')
if page == 2:
    plot_model_history(main_model_train_history, plt_title='Total CO2 Emissions', ax=axs[3, 1])

if page == 2:

    plt.savefig('src/Models/model_checkpoints/train_history/epoch_vs_loss_2.svg')
    # plt.show()
