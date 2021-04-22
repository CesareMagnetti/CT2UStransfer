import torch
import numpy as np
import SimpleITK as sitk

class ToNumpy(object):
    """Convert an itkImage to a ``numpy.ndarray``.
    Converts a itkImage (W x H x D) or numpy.ndarray (D x H X W)
    NOTA: itkImage ordering is different than of numpy and pytorch.
    """

    def __init__(self, outputtype=None):
        self.outputtype = outputtype

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : itkImages in SimpleITK format
            Images to be converted to numpy.
        Returns
        -------
        Numpy nd arrays
        """
        outputs = []
        for idx, _input in enumerate(inputs):
            outputs.append(self._tonumpy(_input,self.outputtype))
        return outputs if idx > 0 else outputs[0]

    def _tonumpy(self, input, outputtype):
        ret = None
        if isinstance(input, sitk.SimpleITK.Image):
            # Extract the numpy nparray from the ITK image
            narray = sitk.GetArrayFromImage(input);
            # The image is now stored as (y,x), transpose it
            ret = np.transpose(narray, [1,2,0])
        elif isinstance(input, np.array):
            # if the input is already numpy, assume it is in the right order
            ret = input

        # overwrite output type if requested
        if self.outputtype is not None:
            ret = ret.astype(outputtype)

        return ret


class ToTensor(object):
    """Convert an itkImage or ``numpy.ndarray`` to tensor.
    Converts a itkImage (N x W x H x D) or numpy.ndarray (N x D x H X W) to a
    torch.FloatTensor of shape (N x D X H X W).
    i.e. itkImage ordering is different than of numpy and pytorch.
    """

    def __call__(self, *inputs):
        """
        Arguments
        ---------
        *inputs : itkImages or numpy.ndarrays
            Images to be converted to Tensor.
        Returns
        -------
        Tensors
        """
        outputs = []
        for idx, _input in enumerate(inputs):
            #_input_is_numpy = isinstance(_input, np.ndarray)
            if isinstance(_input, sitk.SimpleITK.Image):
                # Get numpy array (is a deep copy!)
                _input = sitk.GetArrayFromImage(_input)
                transpose = True
            #print(f'input or converted numpy array type: {_input.dtype}')
            _input = torch.from_numpy(_input.astype(np.double))
            #_input = torch.from_numpy(_input)
            if transpose:
                if len(_input.shape) == 3:
                    _input = _input.permute(2,0,1) #Change size from
                elif len(_input.shape) == 4:
                    _input = _input.permute(0,3,1,2) #Change size from

            # float for backward compatibility ?
            outputs.append(_input.float())
        return outputs if idx > 0 else outputs[0]
