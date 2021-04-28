class Rescale(object):
    def __init__(self, max_intensity=255):

        """
        Rescales input tensor to (0,1) by dividing by maximum possible intensity (default: 255)
           tensor_out = tensor_in/max_intensity (element wise division)
        Arguments
        ---------
        max_intensity : int   (default: 255)

        :TODO: Add support for rescaling multiple tensors to make it consistent with other transforms.
        """

        self.max_intensity = max_intensity

    def __call__(self, tensor):
        """
        Arguments
        ---------
        tensor : Tensor
            Tensor of size (C, H, W, D)

        Returns:
        --------
        Tensor: in the range [0,1]
        """

        return tensor/self.max_intensity
