# SumMe Video Summarization with Spectral Clustering and ResNet50 Features

## Overview

This script implements a video summarization method for the SumMe dataset using spectral clustering and deep features extracted from a pre-trained ResNet50 model. 

The script performs the following steps:

1. **Loads the SumMe dataset:** Reads video files and their corresponding ground truth summaries.
2. **Samples frames:** Extracts frames from each video at a specified sampling rate.
3. **Extracts features:** Uses a pre-trained ResNet50 model to extract deep features from the sampled frames.
4. **Clusters frames:** Applies spectral clustering to group similar frames based on their features.
5. **Selects keyframes:** Chooses representative frames from each cluster to form the video summary. **Note:** Unlike the ultrasound with SonoNet implementation, keyframe selection here is based on the distance to the cluster centroid in the ResNet-50 embedding space, not the spectral embedding space. 
6. **Evaluates the summary:** Calculates the F1 score of the generated summary against the ground truth summary.
7. **Visualizes results:** Displays the sampled frames with highlighted keyframes and ground truth frames for comparison.

## Usage

1. **Update the input and output directories:**
   - Set the `input_dir` variable to the path of the directory where you extracted the SumMe dataset.
   - Set the `output_dir` variable to the path where you want to save the output visualization images.

2. **Run the file:**
   - SumMeDataset.ipynb

## Parameters

You can adjust the following parameters in the script to customize the summarization process:

* `segment_length`: The length of each segment in frames (used for downsampling ground truth).
* `num_clusters`: The number of clusters to use in spectral clustering.
* `sampling_every_ith_frame`: The sampling rate for extracting frames from the video.
* `similarity_threshold`: The similarity threshold used to determine the number of clusters dynamically.
* `debugging`: Set to `True` to enable debugging mode (shows plots interactively instead of saving them).

## Output

The script will print the F1 score for each video in the SumMe dataset and the overall average F1 score. It will also generate visualization images showing the sampled frames with highlighted keyframes and ground truth frames for each video. These images will be saved in the specified `output_dir`.
