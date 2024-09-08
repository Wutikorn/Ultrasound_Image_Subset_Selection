# Ultrasound Grayscale Image Subset Selection

## Overview

This folder contains the code and instructions for performing image subset selection on a grayscale ultrasound image dataset. The dataset is assumed to be in HDF5 format and contain 14 different classes of ultrasound images. The process involves two main steps:

1. **Image Clustering and Subset Selection:** Uses a pre-trained ResNet-50 model (fine-tuned for ultrasound image classification) and spectral clustering to group similar images and select representative keyframes from each cluster.
2. **Image Classification:** Trains a ResNet-18 model on the selected subset of images to evaluate the effectiveness of the subset for image classification.

## Files

* **`best_ultrasound_resnet50.pth`:** Contains the pre-trained weights for the ResNet-50 model used for feature extraction. No action is needed for this file, it will be loaded automatically by the Jupyter Notebook. 
* **`Ultrasound Image Clustering and Classification.ipynb`:** This Jupyter Notebook contains the code for both image clustering/subset selection and the image classification steps.

## Steps

### 1. Image Clustering and Subset Selection

**File:** `Ultrasound Image Clustering and Classification.ipynb` (**Clustering** section)

This section of the notebook performs the following:

1. **Loads the Ultrasound Dataset:** Reads the grayscale ultrasound images and their labels from the specified HDF5 file.
2. **Extracts Features:** Uses the pre-trained ResNet-50 model (`best_ultrasound_resnet50.pth`) to extract features from the images. 
3. **Performs Spectral Clustering:** Applies spectral clustering to group similar images based on their extracted features. 
4. **Selects Keyframes:** From each cluster, selects the image closest to the cluster centroid (using the specified distance metric) as the representative keyframe.
5. **Saves the Subset:** Creates a new HDF5 file containing the selected subset of images and their corresponding labels. 

**How to Run:**

1. **Update HDF5 File Paths:** 
   - Modify the `h5py_input_path` variable to point to the location of your input HDF5 file containing the ultrasound dataset.
   - Modify the `h5py_output_path` variable to specify the location where you want to save the new HDF5 file containing the selected subset.
2. **Adjust Parameters (Optional):** 
   - You can customize parameters like `num_clusters`, `num_key_clusters`, `rbf_gamma` (for the RBF kernel), `dist_to_centroid_metric`, and the `debugging` flag.
3. **Run the Clustering Section:** Execute the cells in the "Clustering" section of the Jupyter Notebook.

### 2. Image Classification

**File:** `Ultrasound Image Clustering and Classification.ipynb` (**Classification** section)

This section of the notebook trains and evaluates a ResNet-18 model on the selected subset of ultrasound images.

**How to Run:**

1. **Use the same HDF5 File Path:**
   - Make sure the `h5py_file_path` variable in the "Classification" section points to the HDF5 file created in the previous step (containing the selected subset). 
2. **Adjust Parameters (Optional):**
   - You can customize training parameters like the number of epochs, batch size, learning rate, etc.
3. **Run the Classification Section:** Execute the cells in the "Classification" section of the Jupyter Notebook. The notebook will output the classification accuracy and per-class metrics (precision, recall, F1-score) on the test set.