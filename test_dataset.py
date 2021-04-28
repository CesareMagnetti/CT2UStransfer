import argparse
from dataset.Horse2Zebra import Horse2Zebra
from utils.helperFunctions import show_batch, show
from torchvision import transforms as torchT
import utils.sitkTransforms as sitkT
import utils.tensorTransforms as tensorT
from torch.utils.data import DataLoader

option = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
option.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, testA, testB, etc)')
option.add_argument('--fold', type=int, default=None, help='which fold to use for validation. if None (default) all data will be used for training.')
option.add_argument('--paired_data', action='store_true', help='flag to use if data provided is paired.')
parser = option.parse_args()

# transform for images
transform = torchT.Compose([sitkT.ToTensor(),
                            torchT.Resize(size=(256,256)),
                            tensorT.Rescale(max_intensity=255)
                           ])

if __name__ == "__main__":
    train_ds = Horse2Zebra(parser, mode="train", transform1 = transform, transform2 = transform)
    #valid_ds = Horse2Zebra(parser, mode="train", isValid=True, transform1 = transform, transform2 = transform)

    train_loader = DataLoader(train_ds, batch_size=32)
    print(len(train_ds))#, len(valid_ds))
    batch = next(iter(train_loader))

    show_batch(batch[0])
    show_batch(batch[1])