from torchvision.utils import make_grid
import numpy as np
from matplotlib import pyplot as plt

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def show_batch(tensors, nrow = 8, denorm = None):
    tensors = tensors.detach().cpu()
    if denorm is not None:
        tensors = denorm(tensors)

    tensors = make_grid(tensors, nrow=nrow, padding=2, normalize=True,
                            range=None, scale_each=False, pad_value=0)
    show(tensors)