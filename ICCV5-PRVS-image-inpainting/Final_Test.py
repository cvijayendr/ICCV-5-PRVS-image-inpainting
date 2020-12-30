from pconv_layer import PConv2D
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from os import listdir
from os.path import isfile, join

def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))

def data_generation(img, dim=(128,128), n_channels=3):
    # Masked_images is a matrix of masked images used as input
    Masked_images = np.empty((1, dim[0], dim[1], n_channels))  # Masked image
    # Mask_batch is a matrix of binary masks used as input
    Mask_batch = np.empty((1, dim[0], dim[1], n_channels))  # Binary Masks
    # y_batch is a matrix of original images used for computing error from reconstructed image
    y_batch = np.empty((1, dim[0], dim[1], n_channels))  # Original image

    
    ## Iterate through random indexes
    for i, idx in enumerate(img):
        image_copy = idx.copy()

        ## Get mask associated to that image
        masked_image, mask = createMask(image_copy)

        Masked_images[i,] = masked_image / 255
        Mask_batch[i,] = mask / 255
        y_batch[i] = idx / 255

    ## Return mask as well because partial convolution require the same.
    return [Masked_images, Mask_batch], y_batch

def createMask(img, dim=(128, 128)):
    ## Prepare masking matrix
    mask = np.full((dim[0], dim[1], 3), 255, np.uint8)  ## White background
    for _ in range(np.random.randint(1, 10)):
        # Get random x locations to start line
        x1, x2 = np.random.randint(1, dim[0]), np.random.randint(1, dim[0])
        # Get random y locations to start line
        y1, y2 = np.random.randint(1, dim[1]), np.random.randint(1, dim[1])
        # Get random thickness of the line drawn
        thickness = np.random.randint(1, 3)  # Change it to 1,9
        # Draw black line on the white mask
        cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), thickness)

    ## Mask the image
    masked_image = img.copy()
    masked_image[mask == 0] = 255

    return masked_image, mask

def testing(image_path,model_path):
    model = keras.models.load_model(model_path,custom_objects={'dice_coef': dice_coef,'PConv2D': PConv2D})
    img = np.empty((1,128,128,3))

    data = np.array(Image.open(image_path).resize((128,128)))
    img[0]=data
    [masked_images, masks],sample_labels = data_generation(img)


    fig, axs = plt.subplots(ncols=4,figsize=(12, 7))

    for i in range(1): 
        inputs = [masked_images[i].reshape((1,)+masked_images[i].shape), masks[i].reshape((1,)+masks[i].shape)]
        impainted_image = model.predict(inputs)
        axs[0].imshow(masked_images[i])
        axs[1].imshow(masks[i])
        axs[2].imshow(impainted_image.reshape(impainted_image.shape[1:]))
        axs[3].imshow(sample_labels[i])
    plt.show()