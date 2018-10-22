"""
Channel augmentation main function.
"""

import numpy as np
from imgaug import augmenters as iaa
from skimage import img_as_ubyte, img_as_float

def augment_channels(images, aug_config):
    """
    Augment each image in images with the channel transformation given in aug_config.
    """
    augmented_images = []
    
    # Instantiate transformations
    if aug_config.do_blur:
        blur = iaa.GaussianBlur(sigma=aug_config.blur_sigma)
    if aug_config.do_edge:
        edge = iaa.EdgeDetect(alpha=1)
    if aug_config.do_contrast:
        contrast = iaa.ContrastNormalization((0.5, 1.5))
    if aug_config.do_convolve:
        convolve = iaa.Convolve(matrix=np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]]))
    if aug_config.do_invert:
        invert = iaa.Invert(1)
    
    # Augment each image
    for im in images:
        augmented = im
        if aug_config.do_blur:
            aug = img_as_float(blur.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))

        if aug_config.do_edge:
            aug = img_as_float(edge.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))

        if aug_config.do_contrast:
            aug = img_as_float(contrast.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))
        
        if aug_config.do_convolve:
            aug = img_as_float(convolve.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))
        
        if aug_config.do_invert:
            aug = img_as_float(invert.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))

        augmented_images.append(augmented)

    return augmented_images
