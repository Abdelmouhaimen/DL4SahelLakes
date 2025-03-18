# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using jupyter.

# This is where you can change which config to use, by replacing 'config_default' with 'my_amazing_config' etc
import config.config_abduh as configuration

# INIT
config = configuration.Configuration().validate()
import preprocessing_crs_fixed
import training
import prediction_abduh
import postprocessing
import pandas as pd

if __name__ == "__main__":

    # PREPROCESSING
    preprocessing_crs_fixed.preprocess_all(config)

    # TRAINING
    #training.train_model(config)

    # PREDICTION
    #prediction_abduh.predict_all(config)

    # POSTPROCESSING
    #postprocessing.postprocess_all(config)


