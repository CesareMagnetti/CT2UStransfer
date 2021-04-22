import torch
from torchvision import transforms as T
import utils.sitkTransforms as sitkT
from utils.dataset import Horse2Zebra
from utils.helperFunctions import show_batch
import os

# transform for images
transform = T.Compose([sitkT.ToTensor(), T.Resize(size=(256,256))])

# set up datasets
data_root = os.path.join(os.getcwd(), "data/horse2zebra")
horsesTrain = Horse2Zebra(root = data_root,
                          kind = "horse", split = "train", transform=transform)
horsesTest = Horse2Zebra(root = data_root,
                          kind = "horse", split = "test", transform=transform)
zebrasTrain = Horse2Zebra(root = data_root,
                          kind = "zebra", split = "train", transform=transform)
zebrasTest = Horse2Zebra(root = data_root,
                          kind = "zebra", split = "test", transform=transform)

show_batch(zebrasTest[:32])