import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import os

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]
def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_image(fname):
    return sitk.ReadImage(fname)

def save_image(itk_img, fname):
    sitk.WriteImage(itk_img, fname)

def gglob(path, regexp=None):
    import fnmatch
    matches = []
    if regexp is None:
        regexp = '*'
    for root, dirnames, filenames in os.walk(path, followlinks=True):
        for filename in fnmatch.filter(filenames, regexp):
            matches.append(os.path.join(root, filename))
    return matches
            

class Horse2Zebra(Dataset):
    def __init__(self, root, kind, split, transform = None):
        """
        Arguments
        ---------
        :param root : string
            Root directory of dataset.
            (downloaded from: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip) 
            should contain the following folders <trainA testA trainB testB> where A refers to horses and B to zebras.
        
        :param kind : string
            Either horse or zebra to retrieve respective data.
        
        :param split : string
            either train or test.

        :param transform : callable, optional (default: None)
            A function/transform that takes in input itk image or Tensor and returns a
            transformed version. E.g, ``torchvision.transforms.RandomCrop()``
        """
        super(Horse2Zebra, self).__init__()

        assert kind.lower() in ("horse", "zebra"), "unknown parameter ``kind``. expected <horse/zebra> got: %s"%kind
        assert split in ("train", "test"), "unknown parameter ``split``. expected <train/test> got: %s"%split

        if kind.lower() == "horse":
            self.root = os.path.join(root, split+"A")
        else:
            self.root = os.path.join(root, split+"B")

        self.filenames = [os.path.realpath(y) for y in gglob(self.root, '*.*') if _is_image_file(y)]      
        self.filenames.sort() #sort files, they will be shuffled by the data loader

        self.transform = transform

    def __getitem__(self, index):
        """
        Arguments
        ---------
        :param index : int
            index position to return the data
        Returns
        -------
        image array (transformed if needed)
        """
        image = load_image(self.filenames[index])
        if self.transform is not None:
            image = self.transform(image)
        return image