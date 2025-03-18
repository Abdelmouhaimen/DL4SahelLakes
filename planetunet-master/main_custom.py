# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using jupyter.

import argparse
import importlib
import preprocessing
import training
import prediction_abduh
import postprocessing
import pandas as pd

def main(config_module_name):
    # Dynamically import the specified configuration module
    config_module = importlib.import_module(config_module_name)
    config = config_module.Configuration().validate()

    # PREPROCESSING
    #preprocessing.preprocess_all(config)

    # TRAINING
    #training.train_model(config)

    # PREDICTION
    prediction_abduh.predict_all(config)

    # POSTPROCESSING
    # postprocessing.postprocess_all(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training, and prediction tasks.')
    parser.add_argument('--config', type=str, required=True, help='The configuration module to use, e.g., config.config_abduh')

    args = parser.parse_args()
    main(args.config)
