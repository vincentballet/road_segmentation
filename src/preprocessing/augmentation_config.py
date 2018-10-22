"""
Image augmentation configuration wrapper.
"""

class ImageAugmentationConfig:
    """
    Encapsulate an augmentation configuration.
    Example:

        config = ImageAugmentationConfig()
        config.rotation([20, 80])
        config.flip()
        config.edge()
        config.blur()
        config.contrast()
    """
    def __init__(self):
        self.do_rotation = False
        self.do_flip = False
        self.augment_channels = False
        self.do_edge = False
        self.do_blur = False
        self.do_contrast = False
        self.do_convolve = False
        self.do_invert = False

    def rotation(self, angles):
        """
        Add rotations.
        """
        self.do_rotation = True
        self.rotation_angles = angles

    def flip(self):
        """
        Add flip transformation.
        """
        self.do_flip = True

    def edge(self):
        """
        Add edge augmentation. 
        """
        self.augment_channels = True
        self.do_edge = True

    def contrast(self):
        """
        Add contrast augmentation.
        """
        self.augment_channels = True
        self.do_contrast = True
    
    def convolve(self):
        """
        Add convolution augmentation.
        """
        self.augment_channels = True
        self.do_convolve = True
    
    def invert(self):
        """
        Add invertion augmentation.
        """
        self.augment_channels = True
        self.do_invert = True
        
    def blur(self, sigma=2):
        """
        Add blur augmentation.
        """
        self.augment_channels = True
        self.do_blur = True
        self.blur_sigma = sigma
