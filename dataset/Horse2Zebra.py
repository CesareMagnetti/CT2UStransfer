class Horse2Zebra(BaseDataset):
    def __init__(self, parser, mode):
        """
        Arguments
        ---------
        :param parser : argparse instance containing all options for the experiments.
                        Please see scripts in the parser folder to inspect available flags.
        :param mode : str, either "train"/"valid"/"test"
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