from dataset.BaseDataset import BaseDataset
import random

class Horse2Zebra(BaseDataset):
    def __init__(self, parser, mode = "train", isValid = False, transform1 = None, transform2 = None):
        """
        Arguments
        ---------
        :param parser : argparse instance containing all options for the experiments.
                        Please see scripts in the parser folder to inspect available flags.
        :param isValid : bool, if this set is to be used as a validation set. (default = False)
        :param transform1 : callable, preprocessing transformation(s) for images in domain1.
        :param transform2 : callable, preprocessing transformation(s) for images in domain2.
        """
        super(Horse2Zebra, self).__init__(parser, mode, isValid)
        # self.rename_domains("Horse", "Zebra") # if you do this make sure you change filenames after downloading them
        self.filenames1, self.filenames2 = self.get_filenames()
        self.size1, self.size2 = len(self.filenames1), len(self.filenames2)
        self.transform1 = transform1
        self.transform2 = transform2

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
        image1 = self.load_image(self.filenames1[index%self.size1]) # make sure index is in range for domain1
        # index 2 could be a random integer for unpaired data or the same index as index1 for paired data
        if self.parser.paired_data:
            index2 = index%self.size2 # make sure in range, paired data
        else:
            index2 = random.randint(0, self.size2-1) # randomize data
        image2 = self.load_image(self.filenames2[index2])

        if self.transform1 is not None:
            image1 = self.transform1(image1)
        if self.transform2 is not None:
            image2 = self.transform2(image2)
        return image1, image2
    
    def __len__(self):
        """
        We have two datasets with different number of images,
        we take a maximum of them as the dataset size.
        """
        return max(self.size1, self.size2)