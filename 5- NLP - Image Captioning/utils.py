import os
import json
import tempfile
import urllib.request

import numpy as np
import h5py
from imageio import imread
from PIL import Image

# Mean and standard deviation for SqueezeNet image preprocessing
SQUEEZENET_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_coco_dataset(base_dir, max_train=None, pca_features=True):
    """
    Load COCO dataset with image features and captions, and optionally apply PCA transformation to features.

    Parameters:
    - base_dir: Directory path where the COCO dataset is stored.
    - max_train: If specified, only a subset of training data is loaded (randomly sampled).
    - pca_features: Boolean flag to decide if PCA features should be loaded for the images.

    Returns:
    - data: A dictionary containing loaded data such as captions, image features, URLs, and vocab.
    """
    print('base dir ', base_dir)
    data = {}

    # Load COCO captions
    caption_file = os.path.join(base_dir, "coco2014_captions.h5")
    with h5py.File(caption_file, "r") as f:
        for k, v in f.items():
            data[k] = np.asarray(v)  # Store captions in dictionary

    # Load training features (PCA or raw)
    if pca_features:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7_pca.h5")
    else:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7.h5")
    with h5py.File(train_feat_file, "r") as f:
        data["train_features"] = np.asarray(f["features"])

    # Load validation features (PCA or raw)
    if pca_features:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7_pca.h5")
    else:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7.h5")
    with h5py.File(val_feat_file, "r") as f:
        data["val_features"] = np.asarray(f["features"])

    # Load vocabulary (word-to-index mapping)
    dict_file = os.path.join(base_dir, "coco2014_vocab.json")
    with open(dict_file, "r") as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v  # Store vocabulary and index mappings

    # Load image URLs for training and validation splits
    train_url_file = os.path.join(base_dir, "train2014_urls.txt")
    with open(train_url_file, "r") as f:
        train_urls = np.asarray([line.strip() for line in f])
    data["train_urls"] = train_urls

    val_url_file = os.path.join(base_dir, "val2014_urls.txt")
    with open(val_url_file, "r") as f:
        val_urls = np.asarray([line.strip() for line in f])
    data["val_urls"] = val_urls

    # Optionally subsample the training data if max_train is specified
    if max_train is not None:
        num_train = data["train_captions"].shape[0]
        mask = np.random.randint(num_train, size=max_train)  # Randomly select a subset of training data
        data["train_captions"] = data["train_captions"][mask]
        data["train_image_idxs"] = data["train_image_idxs"][mask]
        # data["train_features"] = data["train_features"][data["train_image_idxs"]]  # Optionally slice features by indices
    return data

def decode_captions(caption_indices, index_to_word):
    """
    Decode captions from indices into human-readable strings.

    Parameters:
    - caption_indices: Array of caption indices, shape (batch_size, sequence_length) where batch_size is the number of captions
      and sequence_length is the maximum length of a caption.
    - index_to_word: Dictionary mapping index to word (vocabulary).

    Returns:
    - decoded_captions: List of decoded captions (strings) or a single string if input is a single caption.
    """
    is_single_caption = False
    if caption_indices.ndim == 1:
        is_single_caption = True  # Handle single caption as a batch of size 1
        caption_indices = caption_indices[None]  # Convert to batch format

    decoded_captions = []
    batch_size, sequence_length = caption_indices.shape

    for caption_idx in range(batch_size):
        words = []
        for time_step in range(sequence_length):
            word = index_to_word[caption_indices[caption_idx, time_step]]  # Get word from index
            if word != "<NULL>":
                words.append(word)  # Append word if not null token
            if word == "<END>":
                break  # Stop processing if end token is reached
        decoded_captions.append(" ".join(words))  # Join words into a single caption string

    if is_single_caption:
        return decoded_captions[0]  # Return a single caption if input was a single caption
    return decoded_captions

def get_coco_minibatch(coco_data, minibatch_size=100, dataset_split="train"):
    """
    Sample a minibatch from the COCO dataset for training or validation.

    Parameters:
    - coco_data: The loaded COCO dataset dictionary.
    - minibatch_size: The number of samples in the minibatch.
    - dataset_split: The data split to sample from ("train" or "val").

    Returns:
    - sampled_captions: A batch of captions.
    - sampled_image_features: A batch of image features.
    - sampled_image_urls: A batch of image URLs.
    """
    # Get the total number of samples in the specified split
    total_samples = coco_data[f"{dataset_split}_captions"].shape[0]

    # Randomly select a batch of samples
    random_indices = np.random.choice(total_samples, minibatch_size)
    sampled_captions = coco_data[f"{dataset_split}_captions"][random_indices]  # Selected captions
    selected_image_indices = coco_data[f"{dataset_split}_image_idxs"][random_indices]  # Image indices
    sampled_image_features = coco_data[f"{dataset_split}_features"][selected_image_indices]  # Image features
    sampled_image_urls = coco_data[f"{dataset_split}_urls"][selected_image_indices]  # Image URLs

    return sampled_captions, sampled_image_features, sampled_image_urls




def preprocess_image(img):
    """Preprocess an image for SqueezeNet.

    This function scales the image to [0, 1] range by dividing by 255,
    then normalizes the image by subtracting the mean and dividing by the
    standard deviation for each color channel (RGB).

    Args:
        img (numpy.ndarray): The input image to preprocess. It should be in
                              the format (height, width, channels).

    Returns:
        numpy.ndarray: The preprocessed image ready for input into SqueezeNet.
    """
    # Convert image to float32, scale to [0, 1], subtract mean and divide by std
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_IMAGE_MEAN) / SQUEEZENET_IMAGE_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert it back to uint8.

    This function reverses the operations done by preprocess_image, by multiplying
    by the standard deviation and adding the mean, then scaling the image back to
    the [0, 255] range.

    Args:
        img (numpy.ndarray): The preprocessed image to deprocess.
        rescale (bool): Whether to rescale the image pixel values to the [0, 1] range
                        after undoing the normalization (for visualization).

    Returns:
        numpy.ndarray: The deprocessed image, in uint8 format.
    """
    # Undo normalization: multiply by std and add mean
    img = img * SQUEEZENET_IMAGE_STD + SQUEEZENET_IMAGE_MEAN

    # Optionally rescale the image to [0, 1] if desired
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)

    # Clip values to be in [0, 255] and convert to uint8
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """Read an image from a URL.

    This function retrieves an image from the provided URL, saves it temporarily to
    a file, and reads it back into a numpy array.

    Args:
        url (str): The URL of the image to download and load.

    Returns:
        numpy.ndarray: The image data as a numpy array.
    """
    try:
        # Open the URL and read the image data into a temporary file
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()  # Create a temporary file
        with open(fname, "wb") as ff:
            ff.write(f.read())  # Write the image data to the file

        # Read the image into a numpy array
        img = imread(fname)

        # Clean up the temporary file
        os.remove(fname)

        return img  # Return the image data
    except urllib.error.URLError as e:
        print("URL Error: ", e.reason, url)
    except urllib.error.HTTPError as e:
        print("HTTP Error: ", e.code, url)


def load_image(filename, size=None):
    """Load and resize an image from disk.

    This function loads an image from a given file path and resizes it to a specified
    size based on the shortest dimension (height or width). It also supports resizing
    using a specified resampling method.

    Args:
        filename (str): Path to the image file to load.
        size (int, optional): Desired size of the shortest dimension of the image.
                              The other dimension will be scaled proportionally.

    Returns:
        numpy.ndarray: The resized image as a numpy array.
    """
    # Read the image from the file
    img = imread(filename)

    # If a new size is specified, resize the image
    if size is not None:
        orig_shape = np.array(img.shape[:2])  # Get the original height and width
        min_id_x = np.argmin(orig_shape)  # Find the shortest dimension
        scale_factor = float(size) / orig_shape[min_id_x]  # Calculate scale factor
        new_shape = (orig_shape * scale_factor).astype(int)  # Calculate new shape

        # Resize the image using nearest-neighbor resampling (adjust as needed)
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.NEAREST))

    return img  # Return the resized image