import glob
import os
import cv2
from matplotlib.pyplot import imread
import numpy as np
from PIL import Image
import sononet
import torch
import math
import random
from torch.autograd import Variable
from spectral_clustering import process_class  # Import your spectral clustering functions


disclaimer = '''
This is a PyTorch implementation of Efficient Image Subset Selection for Ultrasound Video Frame Analysis via Spectral Clustering and Deep Feature with SonoNet

SonoNet is a fetal standard scan plane classifier and localisation. It scan through videos and output frames into 13 fetal standard scan plane and 1 background class. 
Each class still have thousands of images for a sonographer to manually assess. The efficient image subset selection will dynamically create a highly diverse and informative subset, resulting in 3 to 100 images.

This implementation of image subset selection with SonoNet is the following respoitory:
https://github.com/Wutikorn/Ultrasound_Image_Subset_Selection


The SonoNet is based on:

Baumgartner et al., "Real-Time Detection and Localisation of Fetal Standard
Scan Planes in 2D Freehand Ultrasound", arXiv preprint:1612.05601 (2016)

This SonoNet repository is based on
  https://github.com/baumgach/SonoNet-weights
which provides a theano+lasagne implementation.


The Spectral Clustering code is based on Scikit-learn:
Scikit-learn: Machine Learning in Python
Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E., Journal of Machine Learning Research
https://github.com/scikit-learn/scikit-learn
'''
print(disclaimer)

# Configuration
network_name = 'SN64'   # 'SN16', 'SN32' pr 'SN64'
display_images = False  # Whether or not to show the images during inference
GPU_NR = 0     # Choose the device number of your GPU

# If you provide the original lasagne parameter file, it will be converted to a
# pytorch state_dict and saved as *.pth.
# In this repository, the converted parameters are already provided.c
weights = True
input_size = (288, 224)
# weights = ('/local/ball4916/dphil/SonoNet/SonoNet-weights/SonoNet{}.npz'
#                .format(network_name[2:]))


# Other parameters
crop_range = [(115, 734), (81, 874)]  # [(top, bottom), (left, right)]
input_size = [224, 288]
# --- Set input and output paths here ---
input_video_path = os.environ.get('INPUT_VIDEO_PATH', '/app/data/ultrasound_videos')  # Path to the folder containing .mp4 videos
output_path = os.environ.get('OUTPUT_PATH', '/app/data/keyframes') # Path where output folders will be created

print('input_video_path: ', input_video_path)
print('output_path: ', output_path)

# --------------------------------------
label_names = ['3VV',
               '4CH',
               'Abdominal',
               'Background',
               'Brain (Cb.)',
               'Brain (Tv.)',
               'Femur',
               'Kidneys',
               'Lips',
               'LVOT',
               'Profile',
               'RVOT',
               'Spine (cor.)',
               'Spine (sag.) ']

def calculate_num_clusters(frame_numbers):
    """
    Calculates the number of clusters/keyframes for a given list of frame numbers.

    Args:
        frame_numbers: A list of integers representing frame numbers (assumed to be sorted).

    Returns:
        The total number of clusters/keyframes needed.
    """
    num_clusters = 1  # Start with 1 cluster
    current_segment_length = 1

    for i in range(1, len(frame_numbers)):
        diff = frame_numbers[i] - frame_numbers[i - 1]
        if diff > 20:  # New segment starts
            if current_segment_length > 40:
                num_clusters += math.ceil(current_segment_length * 0.01) 
            elif current_segment_length >= 2:  # Increment only if segment has at least 2 frames
                num_clusters += 1 
            elif current_segment_length == 1:
                num_clusters += 0.5
            current_segment_length = 1 
        else:
            current_segment_length += 1

    # Handle the last segment
    if current_segment_length > 40:
        num_clusters += math.ceil(current_segment_length * 0.01)
    elif current_segment_length >= 2:
        num_clusters += 1

    return int(num_clusters)

def count_frames_per_class(frame_indices):
    """Counts the number of frames and suggests the number of clusters for each class."""

    num_frames = len(frame_indices)

    if num_frames == 0:
        num_clusters = 0  # Set clusters to 0 if no frames
    elif num_frames == 1:
        num_clusters = 1
    else:
        num_clusters = calculate_num_clusters(frame_indices)

    return num_frames, min(num_clusters, 100)

def process_video(video_path, output_path):
    """Processes a single video, extracting frames and classifying them."""

    # Create a new directory for the video's results
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_path, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Create subdirectories for each category
    for label in label_names:
        os.makedirs(os.path.join(video_output_dir, label), exist_ok=True)

    # Dictionary to store frames for each class
    class_frames = {label: [] for label in label_names}

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    print("Step 1 - SonoNet Inference")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 4th frame
        if frame_count % 4 == 0:
            # Preprocess the frame (same as in prepare_inputs)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = imcrop(image, crop_range)
            image = np.array(Image.fromarray(image).resize(input_size, resample=Image.BICUBIC))
            image = np.mean(image, axis=2)
            image_data = np.float32(np.reshape(image, (1, 1, image.shape[0], image.shape[1])))
            mean = image_data.mean()
            std = image_data.std()
            image_data = np.array(255.0 * np.divide(image_data - mean, std), dtype=np.float32)

            # Perform inference
            x = Variable(torch.from_numpy(image_data).cuda())
            outputs = net(x)
            confidence, prediction = torch.max(outputs.data, 1)
            predicted_label = label_names[prediction[0]]

            # Store the frame for later processing
            class_frames[predicted_label].append((frame, frame_count))

        frame_count += 1

    cap.release()


    print("Step 2 - Spectral Clustering and Keyframe Selection")

    for label in class_frames:
        if len(class_frames[label]) > 2000:
            # Sample 2000 frames if more than 2000
            class_frames[label] = random.sample(class_frames[label], 2000)

        # Sort by frame_count (ascending order)
        class_frames[label] = sorted(class_frames[label], key=lambda x: x[1])

    # Process each class using spectral clustering
    for label, frames_and_indices in class_frames.items():
        if frames_and_indices:
            frames, indices = zip(*frames_and_indices)
            # Count frames and get suggested cluster numbers
            num_frames, num_clusters = count_frames_per_class(indices)
            # print(num_frames, num_clusters, len(indices))

            if num_frames <= 2:
                keyframe_indices = indices
                print(f"Class {label} - found {num_frames} frames")
                for i, index in enumerate(keyframe_indices):
                    frame = frames[i]
                    frame_filename = f"{video_name}_keyframe_{original_frame_number:05d}.jpg"
                    output_path = os.path.join(video_output_dir, label, frame_filename)
                    cv2.imwrite(output_path, frame)

            elif num_frames > 0:
                keyframe_indices = process_class(frames, label, num_clusters)
                for i, index in enumerate(keyframe_indices):
                    frame = frames[index]
                    original_frame_number = indices[index]
                    frame_filename = f"{video_name}_keyframe_{original_frame_number:05d}.jpg"
                    output_path = os.path.join(video_output_dir, label, frame_filename)
                    cv2.imwrite(output_path, frame)

    print("Done")


def main():
    global net  # Declare net as global to access it within process_video

    print('Loading network')
    net = sononet.SonoNet(network_name, weights=weights)
    net.to(device)
    net.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Use the first available CUDA device
        print(f'Using CUDA GPU: {torch.cuda.get_device_name(device)}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
        print('Using MPS (Apple Silicon GPU)')
    else:
        device = torch.device("cpu")
        print('No GPU available, using CPU')

    net.to(device)

    run = 0
    # Process all MP4 videos in the directory
    video_paths = glob.glob(os.path.join(input_video_path, '*.MP4'))
    for video_path in video_paths:
        # if video_path == '/vol/bitbucket/wr323/4Wutikorn/videos/iFIND03455_23Feb2016.MP4':
        print(f"Processing video: {video_path}")
        process_video(video_path, output_path)

        run += 1
        if run == 4:
            break


if __name__ == '__main__':
    print("################")
    main()