# Main script to run preprocessing, training or prediction tasks.
# Convert this to a notebook if you are using jupyter.

# This is where you can change which config to use, by replacing 'config_default' with 'my_amazing_config' etc
import config.config_mathilde_gpu0_nasa as configuration

# INIT
config = configuration.Configuration().validate()
import preprocessing
import training
import prediction
import postprocessing

if __name__ == "__main__":

    # PREPROCESSING
    preprocessing.preprocess_all(config)

    # TRAINING
    training.train_model(config)

    # PREDICTION
    prediction.predict_all(config)

    # POSTPROCESSING
  #  postprocessing.postprocess_all(config)
