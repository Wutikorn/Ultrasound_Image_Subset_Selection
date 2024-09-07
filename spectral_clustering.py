import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
import numpy as np
from collections import Counter
from PIL import Image
import numbers
from sklearn.utils.validation import check_array
import os

# --- CONFIGURATION ---
rbf_gamma = 0.5
dist_to_centroid_metric = 'manhattan'
debugging = False  # Set to True to enable debugging outputs
# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# --- MODEL SETUP ---

# Path to your saved model file
model = models.resnet50()  # Create a ResNet50 instance
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 14)  # Match your model's classifier
model.load_state_dict(torch.load('best_ultrasound_resnet50.pth', map_location=device))
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# The Spectral Clustering implementation is from Scikit-learn. Our method use functions from Scikit-learn to access spectral embedding, which otherwise is inaccessible.
# The Spectral embedding is used to calculate distance in spectral embedding to the centroid and for keyframe selection.
# The following functions are from Scikit-learn: _init_arpack_v0, check_random_state, _set_diag, _deterministic_vector_sign_flip, and get_spectral_embedding (affinity matrix and Laplacian matrix)
# https://github.com/scikit-learn/scikit-learn/tree/70fdc843a4b8182d97a3508c1a426acc5e87e980

def _init_arpack_v0(size, random_state):
    """Initialize the starting vector for iteration in ARPACK functions.

    Initialize a ndarray with values sampled from the uniform distribution on
    [-1, 1]. This initialization model has been chosen to be consistent with
    the ARPACK one as another initialization can lead to convergence issues.

    Parameters
    ----------
    size : int
        The size of the eigenvalue vector to be initialized.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator used to generate a
        uniform distribution. If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    v0 : ndarray of shape (size,)
        The initialized vector.
    """
    random_state = check_random_state(random_state)
    v0 = random_state.uniform(-1, 1, size)
    return v0

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.

    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.

    value : float
        The value of the diagonal.

    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.

    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not issparse(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian

def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.

    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.

    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.

    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u

def extract_embeddings(images):
    """Extracts ResNet50 embeddings for a batch of images."""
    embeddings = []
    for img in images:
        # Duplicate channel only before embedding extraction
        img_tensor = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img_tensor)
        embeddings.append(embedding.cpu().squeeze().numpy().astype('double'))
    return embeddings

def duplicate_channel(image_array):
    """Duplicates a single-channel image to create a 3-channel image."""
    if len(image_array.shape) == 2: 
        image_array = np.expand_dims(image_array, axis=0)
    return np.repeat(image_array, 3, axis=0) 

def get_spectral_embedding(all_embeddings, num_clusters):

    # https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/manifold/_spectral_embedding.py

    # 1. Construct affinity matrix (using RBF kernel)
    affinity_matrix = pairwise_kernels(all_embeddings, metric='rbf', gamma=rbf_gamma)

    # 2. Compute graph Laplacian
    norm_laplacian = True
    n_components = 50 #spectral embedding dimension

    laplacian, dd = csgraph_laplacian(
        affinity_matrix, normed=norm_laplacian, return_diag=True
    )

    laplacian = _set_diag(laplacian, 1, norm_laplacian)

    tol = 1e-6
    laplacian *= -1
    v0 = _init_arpack_v0(laplacian.shape[0], random_state=None)
    laplacian = check_array(
        laplacian, accept_sparse="csr", accept_large_sparse=False
    )
    
    _, diffusion_map = eigsh(
        laplacian, k=n_components, sigma=1.0, which="LM", tol=tol, v0=v0
    )
    embeddings = diffusion_map.T[n_components::-1]
    if norm_laplacian:
        # recover u = D^-1/2 x from the eigenvector output x
        embeddings = embeddings / dd

    embeddings = _deterministic_vector_sign_flip(embeddings)

    spectral_embeddings = embeddings[1:n_components].T

    spectral_embeddings = normalize(spectral_embeddings) 
    
    return spectral_embeddings


def process_class(class_images, class_name, num_clusters, debugging=False):
    """
    Performs spectral clustering and keyframe selection for a class 
    and saves the keyframes to an HDF5 file.

    Args:
        class_images (list): List of images for the class, 
                             where each image is a NumPy array with 
                             shape (3, height, width).
        class_name (str): Name of the class.
        output_hdf5_file (h5py.File): HDF5 file object to write the data to.
        debugging (bool, optional): If True, displays debugging plots. 
                                     Defaults to False.
    """

    # print(f"Processing class: {class_name}")
    if len(class_images) - 1 < num_clusters:
        print(f"Not enough images in class {class_name} for clustering.")
        return

    # Extract embeddings for the entire class
    embeddings = extract_embeddings(class_images)

    # Reshape embeddings into 2D array (one embedding per row)
    all_embeddings = np.vstack(embeddings) 

    spectral_embeddings = get_spectral_embedding(all_embeddings, num_clusters)

    # K-Means clustering on spectral_embeddings
    kmeans = KMeans(n_clusters=num_clusters).fit(spectral_embeddings)
    embedding_clusters = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_

    # Keyframe selection

    non_empty_cluster_count = 0
    closest_images = [] 
    closest_image_indices = [] # Store indices of closest images

    for cluster_idx in range(num_clusters):
        cluster_indices = np.where(embedding_clusters == cluster_idx)[0]
        if len(cluster_indices) > 0:
            non_empty_cluster_count += 1
            distances = pairwise_distances(spectral_embeddings[cluster_indices], cluster_centroids[cluster_idx].reshape(1, -1), metric=dist_to_centroid_metric)  # Compute distances in spectral embedding space
            closest_image_idx = cluster_indices[np.argmin(distances)]
            closest_image_indices.append(closest_image_idx)

            # Get the original image (no need to reduce dimensions)
            closest_image = class_images[closest_image_idx] 

            # Save or display the keyframe
            if debugging:
                closest_images.append(closest_image) 


    print(f"Class {class_name} - found {non_empty_cluster_count} keyframes out of {len(class_images)}")

    return closest_image_indices # Return the indices