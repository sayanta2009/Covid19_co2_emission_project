# Dataset information
NUMBER_OF_COUNTRIES = 14
NUMBER_OF_YEARS = 29
TOTAL_NUMBER_OF_FEATURES = 16
COUNTRIES = ['Austria', 'Belgium', 'Bulgaria', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Italy', 'Netherlands', 'Poland', 'Portugal', 'Spain', 'UK']

# model types obtained after running individual pipelines.
FEATURES_BEST_PIPELINE_MODEL = ['cnn', 'cnn', 'cnn_2', 'cnn', 'lstm',
                                'cnn', 'cnn_2', 'gru',
                                'cnn', 'mlp', 'cnn_2', 'cnn',
                                'cnn', 'cnn', 'cnn', 'mlp']
PREDICTION_CASES = ['without covid', 'with covid'] # scenarios for which we are generating the predictions
PREDICT_WITHOUT_COVID_FILE = 'src/Models/prediction_checkpoints/predict_without_covid.pkl'
PREDICT_WITH_COVID_FILE = 'src/Models/prediction_checkpoints/predict_with_covid.pkl'

# Input File paths
ENERGY_FILE = 'Dataset/imputed_data/DataEnergyEmissionsImputed.xlsx'
INDUSTRY_FILE = 'Dataset/imputed_data/DataIndusProcessEmissionsImputed.xlsx'
LAND_USE_FILE = 'Dataset/imputed_data/DataLandUseEmissionsImputed.xlsx'
TRANSPORT_FILE = 'Dataset/imputed_data/DataTransportEmissionsImputed.xlsx'
MAIN_FILE = 'Dataset/imputed_data/DataMainImputedModified.xlsx'

# Output File paths
LSTM_STATS_FILE = 'src/Models/model_checkpoints/lstm/lstm_r2_scores.xlsx'
GRU_STATS_FILE = 'src/Models/model_checkpoints/gru/gru_r2_scores.xlsx'
MLP_STATS_FILE = 'src/Models/model_checkpoints/mlp/mlp_r2_scores.xlsx'
CNN_STATS_FILE = 'src/Models/model_checkpoints/cnn/cnn_r2_scores.xlsx'
CNN_2_STATS_FILE = 'src/Models/model_checkpoints/cnn_2/cnn_2_r2_scores.xlsx'

LSTM_MAIN_MODEL_STRUCTURE = 'src/Models/model_checkpoints/lstm/lstm_main_model_structure.pdf'
LSTM_FEATURE_MODEL_STRUCTURE = 'src/Models/model_checkpoints/lstm/lstm_feature_model_structure.pdf'
GRU_FEATURE_MODEL_STRUCTURE = 'src/Models/model_checkpoints/gru/gru_main_model_structure.pdf'
CNN_MAIN_MODEL_STRUCTURE = 'src/Models/model_checkpoints/cnn/cnn_main_model_structure.pdf'
CONV2D_MAIN_MODEL_STRUCTURE = 'src/Models/model_checkpoints/cnn_2/cnn2d_main_model_structure.pdf'
MLP_MAIN_MODEL_STRUCTURE = 'src/Models/model_checkpoints/mlp/mlp_main_model_structure.pdf'

# Train Parameters
LOOK_BACK_TIME_STEPS = 3

# Common execute parameters
EPOCHS = 2000
BATCH_SIZE = 5
TEST_SPLIT_SIZE = 0.25
VALIDATION_SPLIT_SIZE = 0.3
RELU_ACTIVATION = 'relu'
LINEAR_ACTIVATION = 'linear'

# Lstm initial model parameters
HIDDEN_LAYER_NEURONS = 200
DROPOUT_RATE = 0.2
ADAM_OPTIMIZER = 'adam'
RMS_OPTIMIZER = 'rmsprop'

HIDDEN_LAYER_NEURONS_MAIN_MODEL = 250

# Running pipeline
MODEL_TYPES = ['CNN', 'CNN_2', 'MLP', 'LSTM', 'GRU']
STATS_FILES = [CNN_STATS_FILE, CNN_2_STATS_FILE, MLP_STATS_FILE, LSTM_STATS_FILE, GRU_STATS_FILE]

# Output graph
OUTPUT_GRAPH_PATH = 'src/Models/prediction_checkpoints/predict.png'
