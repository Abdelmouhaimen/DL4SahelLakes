# Adaptation of a CNN U-Net Algorithm for Lake Recognition in Landsat Images

## 1. Downloading and Preprocessing Images

### 1.1 Preprocessed Images

**Defining the Area of Interest:**  
The first step in creating ground truth data is selecting regions from **Sentinel-2 tiles** that include water surfaces. The selected areas must be from **UTM zones 29, 30, and 31**.  
The **Universal Transverse Mercator (UTM)** projection is a conformal map projection system that divides the Earth into **60 zones**, each spanning **6 degrees** in longitude. This results in **120 different projections** (60 for the Northern Hemisphere and 60 for the Southern Hemisphere).

**Water Body Polygons:**  
To increase dataset diversity, we delineate water bodies of various sizes and shapes. This process is complex since some objects may not clearly be water surfaces.  
For annotation, we use **QGIS** to **manually draw polygons** outlining lakes and rectangles defining the image regions used during training.

> *Note: Not all images within a tile are annotated due to the complexity of labeling every single lake.*

### 1.2 Image Downloading

Before running the algorithm, satellite tile images must be downloaded manually from:  
🔗 [Earth Explorer](https://earthexplorer.usgs.gov/)

Alternatively, the process can be automated, as demonstrated in **Mathilde's script** `/mnt/md0/mathilde/sentinel2/theia_download.py`.  
The dataset, including NDWI and MNDWI bands, is stored in `/mnt/md0/abduh`.

---

## 2. Code Structure

The code is structured around **three main stages**:

1. **Preprocessing (`preprocessing.py`)** - Extracts training zones and creates temporary preprocessed frames.
2. **Training (`training.py`)** - Trains the **CNN U-Net model** on the dataset.
3. **Prediction (`prediction.py`)** - Applies the trained model to detect lakes in new satellite images.

These scripts are executed from `main.py` and rely on methods within the `/core/` directory.  
The **general configuration** is stored in `/config/` and initialized in `main.py`.

### Setting Up the Python Environment

To set up the environment using **conda**, run:

```sh
conda activate tf
```

---

## 2.1 Preprocessing

During preprocessing, training areas are extracted from satellite images and stored as **temporary preprocessed frames**.  
Each frame consists of:
- **Image channels**
- **Two annotation channels** (labels & boundary weights)

---

## 2.2 Model Training

The **U-Net model** is trained using the preprocessed dataset and stored in an `.h5` file.  
Metadata related to the model configuration is also saved.

To monitor training progress, logs are recorded in the `/logs/` directory.  
Logs can be visualized using **TensorBoard** with the command:

```sh
tensorboard --logdir [log_path]
```

---

## 2.3 Prediction

During inference, the trained model is applied to new satellite images to **detect water bodies**.  
Predictions are **stored as single-channel raster images**.

### Improvements in Prediction:
The original prediction code was modified to **handle nodata values**.  
In satellite image analysis, **nodata regions** can sometimes be misclassified as water, leading to **false positives**.

To resolve this issue, we added a step in `predict_using_model` to replace **nodata pixels with zero (0)** in the predictions.  
This ensures that nodata areas are correctly excluded from final results.

---

## 3. Configuration

Before running `main.py`, the required parameters must be set in `config_default.py`.  
For example, modifying `prediction_batch_size=64/32` for `patch_size=64` speeds up inference.

You can create **custom configuration files** for different datasets or use cases.  
These configurations can be **specified when launching `main.py`**.  

A separate script, **`main_custom.py`**, allows specifying the configuration file as an argument, enabling batch execution of multiple predictions or training runs.

### **Example Configuration File (`config_abduh.py`):**

Key parameters include:

- **`trained_model_path`** → Path to the trained model for inference.
- **`prediction_output_dir`** → Directory where predictions will be saved.
- **`prediction_stride`** → Distance between consecutive patches during prediction.
- **`prediction_patch_size`** → Size of the image patches processed by the model.
- **`prediction_batch_size`** → Number of patches processed simultaneously.
- **`channels_used`** → Boolean list specifying the **RGB channels** to use.

Each parameter is documented within the configuration file.

---

## 3.2 Running the Pipeline

Executing `main.py` runs the **entire pipeline**:  
- **Preprocessing**
- **Training**
- **Prediction**

By default:
- **Training uses the latest preprocessed data**
- **Prediction applies the most recent trained model**
- **Post-processing refines the most recent predictions**

You can also specify an **existing model checkpoint** for continued training using:

```python
config.continue_model_path = "path/to/trained_model.h5"
```

---

## 4. Model Evaluation

To evaluate the model's performance, the `testing_stride.py` script is used.  
The testing workflow follows the same **configuration principles** as training and prediction.

Key steps:
1. Specify the model path → `self.trained_model_path`
2. Set the preprocessed dataset path → (which contains the annotations and images)
3. Modify **`prediction_stride`** to optimize performance.

---

## 5. Internship Progress & Reports

📄 **Weekly progress and reports** are documented in Progress_CR.pdf


---

## 6. Summary

- **Objective:** Develop a **CNN U-Net model** for detecting **lakes in Landsat images**.
- **Tools Used:** Python, TensorFlow, QGIS, GDAL, Sentinel-2, Landsat.
- **Dataset:** Sentinel-2 satellite images (UTM zones 29, 30, 31).
- **Model Improvements:**
  - **Nodata handling** to eliminate false positives.
  - **QGIS integration** for dataset annotation and visualization.
  - **Patch-based segmentation** for processing large satellite images.
  - **Automated dataset retrieval** from **Earth Explorer**.


