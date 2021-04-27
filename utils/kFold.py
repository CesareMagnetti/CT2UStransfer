'''
creates K text files (1 for each fold) containing filenames of data. This is useful to divide train
data into K-1 folds for training and 1 of validation, which can then be cross-validated.

example of terminal run for a ten fold of training data (random seed allows reproducibility when shuffling filenames):

source env/bin/activate
python kFold.py -r path/to/train_data/dir -K 10 --random_seed 1

Author: Cesare Magnetti
Contact: cesare.magnetti98@gmail.com
'''

import os
import random
import argparse

parser = argparse.ArgumentParser(description='creates K text files (1 for each fold) containing data filenames.')
parser.add_argument('--root','-r', help='directory containing data')
parser.add_argument('--random_seed', '-s', default = 1, type=int, help='random seed for reproducibility')
parser.add_argument('--kFolds', '-K', default = 10, type=int, help='number of folds to split data into')
parser.add_argument('--serial', action='store_true', help='flag to not shuffle data before creating folds. (shuffles data if not parsed)')
args = parser.parse_args()

def gglob(path, regexp=None):
    """Recursive glob
    """
    import fnmatch
    import os
    matches = []
    if regexp is None:
        regexp = '*'
    for root, dirnames, filenames in os.walk(path, followlinks=True):
        for filename in fnmatch.filter(filenames, regexp):
            matches.append(os.path.join(root, filename))
    return matches

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"] # some general image extensions of our datasets
def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

if __name__ == '__main__':

    #get the filenames, sort them to ensured that shuffling with a random seed will give same shuffled list
    filenames = [os.path.realpath(y) for y in gglob(args.root, '*.*') if _is_image_file(y)]
    filenames.sort()

    #set random seed
    random.seed(args.random_seed)
    #randomly shuffle the filenames
    if not args.serial:
        random.shuffle(filenames)

    N=len(filenames)
    n=int(N/args.kFolds) #ignore a couple of files to get equal sized folds (always ignoring less than K files)

    #check if fold directory exists, else create it
    if not os.path.exists(os.path.join(args.root, "folds")):
        os.mkdir(os.path.join(args.root, "folds"))

    #write the K folds text files
    for i in range(args.kFolds):
        with open(args.root+"/folds/{}.txt".format(i+1), 'w') as f:
            start_index = n*i
            end_index = n*(i+1)
            for item in filenames[start_index:end_index]:
                 f.write("%s\n" % item)
