"""
Run this file for training or inference purpose
"""

from optparse import OptionParser
from src.Models.pipeline import *
from src.Models.prediction import *


def get_args():
    """
    Gets the arguments from the command line.
    """
    parser = OptionParser('AMI_Pipeline_Group15:')
    t_desc = 'Boolean field, True if we want to run the Pipeline in execute mode or evaluation mode (as of saved ' \
             'checkpoints)'
    m_desc = 'Boolean field, True if we want to consider the Main Model pipeline or pipelines of features'
    m_desc = 'Boolean field, True if we want to consider the Main Model pipeline or pipelines of features'
    parser.add_option("--train", action="store_true", dest="verbose", help=t_desc)
    parser.add_option("--main_model", action="store_true", dest="verbose", help=m_desc)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline = False
    to_train = False  # True for execute, False for evaluate
    main_model = False  # True for main model, False for first level features
    r2_scores = dict()  # stores the r2 score
    (opts, args) = get_args()
    if args:
        to_train = args[0]
        main_model = args[1]
        if main_model == 'True':
            main_model = True
        elif main_model == 'False':
            main_model = False
        else:
            main_model = False

        if to_train == 'True':
            to_train = True
        elif to_train == 'False':
            to_train = False
        else:
            to_train = False

    if run_pipeline:
        if not main_model:
            # to execute or evaluate for all first level features
            for index, model_type in enumerate(MODEL_TYPES):
                r2_scores = dict()  # stores the r2 score and RMSE error
                for sector in range(4):
                    pipeline = Pipeline(sector, model_type, train=to_train)
                    r2_scores.update(pipeline.execute())
                # to write LSTM r2 scores into excel
                if to_train:
                    write_stats_to_excel(r2_scores, STATS_FILES[index], 'first_level_features_train', to_train)
                else:
                    write_stats_to_excel(r2_scores, STATS_FILES[index], 'first_level_features_test', to_train)
        else:
            # to execute or evaluate for the main co2 emission model
            for index, model_type in enumerate(MODEL_TYPES):
                r2_scores = dict()  # stores the r2 score
                pipeline = Pipeline(4, model_type, train=to_train)
                r2_scores.update(pipeline.execute())
                if to_train:
                    write_stats_to_excel(r2_scores, STATS_FILES[index], 'main_model_train', to_train)
                else:
                    write_stats_to_excel(r2_scores, STATS_FILES[index], 'main_model_test', to_train)
    else:
        # This section is executed for the prediction part.
        prediction = Prediction('Germany', [40, 0, 0, 30, 0, 20, 0, 0, 5, 0, 0, 0, 0, 0, 10])
        prediction.create_visual_graph()
