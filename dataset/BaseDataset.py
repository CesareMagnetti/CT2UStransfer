import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod, ABCMeta
import SimpleITK as sitk
import os

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"] # some general image extensions of our datasets

class BaseDataset(Dataset, ABC):
    """
    Base abstract class for all datasets. 
    Future datasets MUST inherit from this class and implement the following class methods:
        - __init__(): initialization function for the specific dataset
        - __getitem__(): get a single data sample. may be (data, label) pair or simply (data)
        - __len__(): return number of samples in dataset
    """
    __metaclass__  = ABCMeta

    @abstractmethod
    def __init__(self, parser):
        self.parser = parser
        self.root = parser.dataroot

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("children subclasses of ``BaseDataset`` must implement ``__getitem__`` method!")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("children subclasses of ``BaseDataset`` must implement ``__len__`` method!")

    @staticmethod
    def _is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    @staticmethod
    def load_image(self, fname):
        return sitk.ReadImage(fname)

    @staticmethod
    def save_image(self, itk_img, fname):
        sitk.WriteImage(itk_img, fname)

    @staticmethod
    def retrieve_all_files(path, regexp=None):
        import fnmatch
        matches = []
        if regexp is None:
            regexp = '*'
        for root, dirnames, filenames in os.walk(path, followlinks=True):
            for filename in fnmatch.filter(filenames, regexp):
                matches.append(os.path.join(root, filename))
        return matches