import config.config_test as configuration

# INIT
config = configuration.Configuration().validate()

import os
import json
import glob
import time
import shutil
import pandas as pd
from itertools import product
from rasterio.windows import Window, bounds

import math
import h5py
import rasterio
import numpy as np
from tqdm import tqdm

import tensorflow as tf

from core.frame_info import FrameInfo, image_normalize
from core.optimizers import get_optimizer
from core.losses import accuracy, dice_coef, dice_loss, specificity, sensitivity, get_loss, true_positives, false_positives, true_negatives, false_negatives, precision, f1score


def load_model(config):
    """Load a saved Tensorflow model into memory"""

    # Load and compile the model
    model = tf.keras.models.load_model(config.trained_model_path, compile=False)
    model.compile(optimizer=get_optimizer(config.optimizer_fn), loss=get_loss(config.loss_fn, config.tversky_alphabeta))

    return model


def load_model_info():
    """Get and display config of the pre-trained model. Store model metadata as json in the output prediction folder."""

    # If no specific trained model was specified, use the most recent model
    if config.trained_model_path is None:
        model_fps = glob.glob(os.path.join(config.saved_models_dir, "*.h5"))
        config.trained_model_path = sorted(model_fps, key=lambda t: os.stat(t).st_mtime)[-1]

    # Print metadata from model training if available
    print(f"Loaded pretrained model from {config.trained_model_path} :")
    with h5py.File(config.trained_model_path, 'r') as model_file:
        if "custom_meta" in model_file.attrs:
            custom_meta = json.loads(model_file.attrs["custom_meta"].decode("utf-8"))
            print(custom_meta, "\n")

            # Read patch size to use for prediction from model
            if config.prediction_patch_size is None:
                config.prediction_patch_size = custom_meta["patch_size"]

def get_all_frames():
    """Get all pre-processed frames which will be used for training."""

    # If no specific preprocessed folder was specified, use the most recent preprocessed data
    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(config.preprocessed_base_dir,
                                               sorted(os.listdir(config.preprocessed_base_dir))[-1])

    # Get paths of preprocessed images
    image_paths = [os.path.join(config.preprocessed_dir, fn) for fn in os.listdir(config.preprocessed_dir) if
                   fn.endswith(".tif")]
    print(f"Found {len(image_paths)} input frames in {config.preprocessed_dir}")

    # Build a frame for each input image
    frames = []
    frames_nodataval = []
    for im_path in tqdm(image_paths, desc="Processing frames"):

        # Open preprocessed image
        preprocessed = rasterio.open(im_path)

        # Sve a nodata value which will be used for normalizing
        frames_nodataval.append(preprocessed.nodatavals[0])
        preprocessed = preprocessed.read()


        # Get image channels   (last two channels are labels + weights)
        image_channels = preprocessed[:-2, ::]

        # Transpose to have channels at the end
        image_channels = np.transpose(image_channels, axes=[1, 2, 0])

        # Get annotation and weight channels
        annotations = preprocessed[-2, ::]
        weights = preprocessed[-1, ::]

        # Create frame with combined image, annotation, and weight bands
        frames.append(FrameInfo(image_channels, annotations, weights))

    return frames, frames_nodataval

def get_patch_offsets(image, patch_width, patch_height, stride):
    """Get a list of patch offsets based on image size, patch size and stride."""
    height, width = image.shape[:2]
    offsets = list(product(range(0, width, stride), range(0, height, stride)))
    return offsets

def add_to_result(res, prediction, row, col, he, wi, operator='MAX'):
    """Add results of a patch to the total results of a larger area.

    The operator can be MIN (useful if there are too many false positives), or MAX (useful for tackling false negatives)
    """
    curr_value = res[row:row + he, col:col + wi]
    new_predictions = prediction[:he, :wi]
    if operator == 'MIN':
        curr_value[curr_value == -1] = 1  # For MIN case mask was initialised with -1, and replaced here to get min()
        resultant = np.fmin(curr_value, new_predictions)
    elif operator == 'MAX':
        resultant = np.fmax(curr_value, new_predictions)
    else:  # operator == 'REPLACE':
        resultant = new_predictions
    res[row:row + he, col:col + wi] = resultant
    return res


def predict_using_model(model, batch, batch_pos, mask, operator):
    """Predict one batch of patches with tensorflow, and add result to the output prediction."""
    tm = np.stack(batch, axis=0)
    prediction = model.predict(tm)
    for i in range(len(batch_pos)):
        col, row, wi, he = batch_pos[i]
        p = np.squeeze(prediction[i], axis=-1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        mask = add_to_result(mask, p, row, col, he, wi, operator)
    return mask

def predict_and_merge(frames, frames_nodataval):
    """Generate overlapping patches, predict and merge them."""

    # Load the model
    model = load_model(config)
    
    merged_predictions = []
    merged_annotations = []
    patch_width, patch_height = config.prediction_patch_size
    stride = config.prediction_stride

    for i, frame in enumerate(tqdm(frames,position=0, desc="Predicting frames")):
        img = frame.img
        height, width = img.shape[:2]
        annotations = frame.annotations

        # Get list of patch offsets to predict for this image
        offsets = get_patch_offsets(img, patch_width, patch_height, stride)
        
        # Initialise mask to zeros, or -1 for MIN operator
        mask = np.zeros((height, width), dtype=np.float32)
        if config.prediction_operator == "MIN":
            mask = mask - 1
        

        batch, batch_pos = [], []
        big_window = Window(0, 0, width, height)
        for col_off, row_off in tqdm(offsets, position=1, leave=False, desc=f"Predicting {len(offsets)}/"
                                 f"{math.ceil(width/stride)*math.ceil(height/stride)} patches..."):
            # Initialise patch with zero padding in case of corner images. size is based on number of channels
            patch = np.zeros((patch_height, patch_width, np.sum(config.channels_used)))

            # Load patch window from image, reading only necessary channels
            patch_window = Window(col_off=col_off, row_off=row_off, width=patch_width, height=patch_height).intersection(big_window)

            temp_im = img[patch_window.row_off:patch_window.row_off+patch_window.height,
                          patch_window.col_off:patch_window.col_off+patch_window.width, :]
            
            # Normalize the image along the width and height i.e. independently per channel. Ignore nodata for normalization
            temp_im = image_normalize(temp_im, axis=(0, 1), nodata_val=frames_nodataval[i])
            
            # Add to batch list
            patch[:patch_window.height, :patch_window.width] = temp_im
            batch.append(patch)
            batch_pos.append((patch_window.col_off, patch_window.row_off, patch_window.width, patch_window.height))

            # Predict one batch at a time
            if len(batch) == config.prediction_batch_size:
                mask = predict_using_model(model, batch, batch_pos, mask, config.prediction_operator)
                batch, batch_pos = [], []
                
        # Run once more to process the last partial batch (when image not exactly divisible by N batches)
        if batch:
            mask = predict_using_model(model, batch, batch_pos, mask, config.prediction_operator)

        # For non-float formats, convert predictions to 1/0 with a given threshold
        if config.output_dtype != "float32":
            mask[mask < config.prediction_threshold] = 0
            mask[mask >= config.prediction_threshold] = 1

        merged_predictions.append(mask)
        merged_annotations.append(annotations)

    return np.array(merged_predictions), np.array(merged_annotations)


'''def compute_metrics(predictions, annotations, metrics):
    """Compute the metrics on the merged predictions and annotations."""
    results = {metric.__name__: [] for metric in metrics}
    for pred, ann in zip(predictions, annotations):
        for metric in metrics:
            metric_value = metric(ann[...,np.newaxis], pred[...,np.newaxis])
            #if isinstance(metric_value, tf.Tensor):
                #metric_value = tf.reduce_mean(tf.cast(metric_value, dtype=tf.float32))
            results[metric.__name__].append(metric_value.numpy())
    return {metric: np.mean(values) for metric, values in results.items()}'''

def compute_metrics(predictions, annotations, metrics):
    smooth=0.0000001
    """Compute the metrics on the merged predictions and annotations."""
    results = {metric_name: [] for metric_name in metrics}
    tp, tn, fp, fn = 0,0,0,0
    for y_pred, y_t in zip(predictions, annotations):
        tp += np.sum(y_t * y_pred)
        tn += np.sum((1 - y_t) * (1 - y_pred))
        fp += np.sum((1 - y_t) * y_pred)
        fn += np.sum((y_t) * (1 - y_pred))
    total = (tp+tn+fp+fn)
    acc = (tp + tn) / total
    recall = tp / (tp+fn)
    spec = tn / (tn+fp)
    pre = tp / (tp+fp)
    f1_score = (2. * tp) / (2.*tp + fp + fn)

    results["specificity"] = [spec]
    results["sensitivity"] = [recall]
    results["accuracy"] = [acc]
    results["precision"] = [pre]
    results["f1score"] = [f1_score]
    results["true_positives"] = [tp / total]
    results["true_negatives"] = [tn / total]
    results["false_positives"] = [fp / total]
    results["false_negatives"] = [fn / total]
    return results

def test_model(conf):
    """Create and train a new model"""
    global config
    config = conf

    # Get all training frames
    frames, frames_nodataval = get_all_frames()

    # Load model info
    load_model_info()

    # Predict and merge patches
    merged_predictions, merged_annotations = predict_and_merge(frames, frames_nodataval)

    # Compute metrics on merged predictions
    print("Computing metrics on merged predictions")
    #metrics = [dice_coef, dice_loss, specificity, sensitivity, accuracy, precision, true_positives, true_negatives, false_positives, false_negatives, f1score]
    metrics = ["dice_coef", "dice_loss", "specificity", "sensitivity", "accuracy", "precision", "true_positives", "true_negatives", "false_positives", "false_negatives", "f1score"]
    results = compute_metrics(merged_predictions, merged_annotations, metrics)
    #print(results)
    print("f1score: {:0.2f}\n".format(results["f1score"][0]*100))
    print("sensitivity: {:0.2f}\n".format(results["sensitivity"][0]*100))
    print("precision: {:0.2f}\n".format(results["precision"][0]*100))


    # Save results to CSV
    name_csv = f"{config.model_name}"
    df = pd.DataFrame.from_dict(results, orient='index', columns=[name_csv])
    df.to_csv(f"{config.test_dir}{config.model_name}.csv", decimal=',')

if __name__ == "__main__":
    test_model(config)
