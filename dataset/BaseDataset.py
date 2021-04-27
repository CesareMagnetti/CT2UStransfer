import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod, ABCMeta
import SimpleITK as sitk
import os
import fnmatch

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"] # some general image extensions of our datasets

class BaseDataset(Dataset, ABC):
    """
    Base abstract class for all datasets. Our data set will consits of unpaired images of two domains
    respectively identified as A and B. We assume data to be downloaded and stored at ``parser.dataroot``
    and the root folder should contains subfolders <trainA/trainB/testA/testB>. Specific children dataset
    may modify the identifiers A and B to more specific labels (i.e. US and CT).

    Future datasets MUST inherit from this class and implement the following class methods:
        - __init__(): initialization function for the specific dataset.
        - __getitem__(): get a single data sample. may be (data, label) pair or simply (data).
        - __len__(): return number of samples in dataset.

    The class takes an argparse instance that contains all options and flags of the experiments. 
    See the parser folder for more info about these options. 
    """
    __metaclass__  = ABCMeta

    @abstractmethod
    def __init__(self, parser):
        self.root = parser.dataroot
        self.ID1, self.ID2 = "A", "B"
        self.name1, self.name2 = parser.mode + self.ID1, parser.mode + self.ID2
        self.path1, self.path2 = os.path.join(self.root, self.name1), os.path.join(self.root, self.name2)
        self.parser = parser


    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("children subclasses of ``BaseDataset`` must implement ``__getitem__`` method!")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("children subclasses of ``BaseDataset`` must implement ``__len__`` method!")

    def rename_domains(self, ID1, ID2):
        self.ID1 = ID1, self.ID2 = ID2
        self.name1, self.name2 = self.parser.mode + self.ID1, self.parser.mode + self.ID2
        self.path1, self.path2 = os.path.join(self.root, self.name1), os.path.join(self.root, self.name2)

    def get_filenames(self):
        filenames1, filenames2 = [], []
        if self.parser.fold is not None:
            # ensure folds were created
            assert os.path.exists(os.path.join(self.path1, "folds")), "folder ``folds`` not found at: %s\n"%self.path1\
                                                                        "make sure you have run ``utils/kFold.py`` to fold your data."

            assert os.path.exists(os.path.join(self.path2, "folds")), "folder ``folds`` not found at: %s\n"%self.path2\
                                                                        "make sure you have run ``utils/kFold.py`` to fold your data."

            folds_text_files1 = os.listdir(os.path.join(self.path1, "folds"))
            folds_text_files2 = os.listdir(os.path.join(self.path2, "folds"))

            # both datasets must have been folded equally
            assert len(folds_text_files1) == len(folds_text_files2), "make sure that the two datasets have been folded using the same"\
                                                                    " -K flag in utils/kFold.py."
            
            # keep the appropriate folds based on the dataset mode.
            if self.parser.mode == "train":
                # simply remove the validate fold from the files
                folds_text_files1.remove("{}.txt".format(self.parser.fold))
                folds_text_files2.remove("{}.txt".format(self.parser.fold))
            else: 
                # only keep the validate fold
                folds_text_files1 = ["{}.txt".format(self.parser.fold),]
                folds_text_files2 = ["{}.txt".format(self.parser.fold),]

            # retrieve filenames
            for fname1, fname2 in zip(folds_text_files1, folds_text_files2):
                with open(fname1) as f1:
                    for line in f1:
                        if os.path.exists(line.strip()):
                            filenames1.append(line.strip())
                with open(fname2) as f2:
                    for line in f2:
                        if os.path.exists(line.strip()):
                            filenames2.append(line.strip())
        
        else:
            filenames1 = self.retrieve_all_files(self.path1)
            filenames2 = self.retrieve_all_files(self.path2)

        # make sure we only keep image files
        filenames1 = [y for y in filenames1 if self._is_image_file(y)]
        filenames2 = [y for y in filenames2 if self._is_image_file(y)]

        return filenames1, filenames2

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
        matches = []
        if regexp is None:
            regexp = '*'
        for root, dirnames, filenames in os.walk(path, followlinks=True):
            for filename in fnmatch.filter(filenames, regexp):
                matches.append(os.path.join(root, filename))
        return matches