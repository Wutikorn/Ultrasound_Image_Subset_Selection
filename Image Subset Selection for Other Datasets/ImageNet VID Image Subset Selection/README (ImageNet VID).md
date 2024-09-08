# ImageNet VID Image Subset Selection

## Overview

This folder contains the code and instructions for performing image subset selection on the ImageNet VID dataset. The process involves three main steps:

1. **Pre-processing:** Converting the original ImageNet VID video data into individual frames and organizing them into separate folders.
2. **Image Clustering and Subset Selection:** Using a pre-trained ResNet-50 model to extract image's feature and perform spectral clustering to group similar frames and select representative keyframes from each cluster. **Note:** The distance to the centroid for keyframe selection is calculated in the ResNet-50 embedding space, unlike in `Ultrasound_Image_Subset_Selection_with_SonoNet` where it's calculated in the spectral embedding space.
3. **Image Classification:** Training a ResNet-18 model on the selected subset of images to evaluate the effectiveness of the subset for image classification.

## Steps

### 1. Pre-processing: Video to Frame Conversion

**File:** `ImagenetVid Pre-Processing Video-to-frame Separate Folder.ipynb`

This Jupyter Notebook converts the ImageNet VID videos (in MP4 format) into individual frames (JPG format) and saves them into separate folders. It samples frames at a rate of 6 frames per second (FPS).

**How to Run:**

1. **Update input and output directory paths:**
   - Modify the `input_dir_path_train`, `output_dir_path_train`, `input_dir_path_val`, and `output_dir_path_val` variables to point to the correct locations of your ImageNet VID dataset and the desired output directory for the frames.
2. **Run all cells:** Execute all the cells in the Jupyter Notebook to perform the video-to-frame conversion.

### 2. Image Clustering and Subset Selection

**File:** `ImageNetVID Image Clustering Pre-Trained-CNN and Spectral.ipynb`

This Jupyter Notebook performs image clustering and subset selection using a pre-trained ResNet-50 model and spectral clustering. It extracts features from the frames generated in the previous step, clusters them, and selects representative keyframes based on their distance to the cluster centroids in the ResNet-50 embedding space.

**How to Run:**

1. **Update directory paths:** 
   - Modify the `base_dir`, `img_save_base_dir`, and `result_save_dir` variables to point to the correct input (frames from step 1), output (for selected subsets), and results directories.
2. **Adjust parameters:** You can customize the number of clusters (`num_clusters`), the number of key clusters to select from (`num_key_clusters`), and the `debugging` flag (to enable/disable visualization). 
3. **Run all cells:** Execute all the cells in the Jupyter Notebook to perform image clustering and subset selection.

### 3. Image Classification

**File:** `ImageNetVID Image Classificaion Spectral Pre-trained Model.ipynb`

This Jupyter Notebook trains a ResNet-18 model on the selected subset of images generated in the previous step. This helps to evaluate the effectiveness of the subset for image classification.

**How to Run:**

1. **Update dataset paths:** 
   - Modify the `train_data` and `val_data` paths to point to the training and validation subsets generated in step 2.
2. **Adjust parameters (optional):** You can customize the training parameters such as the number of epochs, batch size, learning rate, etc.
3. **Run all cells:** Execute all the cells in the Jupyter Notebook to train the ResNet-18 model and evaluate its performance on the validation set.
