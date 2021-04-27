import argparse
from dataset.Horse2Zebra import Horse2Zebra
from utils.helperFunctions import show_batch
from torchvision import transforms as T
import utils.sitkTransforms as sitkT

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, testA, testB, etc)')
parser.add_argument('--fold', type=int, default=None, help='which fold to use for validation. if None (default) all data will be used for training.')
parser.parse_args()

# transform for images
transform = T.Compose([sitkT.ToTensor(), T.Resize(size=(256,256))])

if __name__ == "__main__":
    print(parser.dataroot)
    train_ds = Horse2Zebra(parser, mode="train", transform1 = transform, transform2 = transform)
    print(len(train_ds))
    batch = train_ds[:32]
    show_batch(batch[0])
    show_batch(batch[1])