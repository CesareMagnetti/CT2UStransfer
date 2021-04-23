import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
from matplotlib import pyplot as plt

def show(img):
    npimg = img.detach().cpu().numpy().squeeze()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def show_batch(tensors, nrow = 8, denorm = None):
    tensors = tensors.detach().cpu()
    if denorm is not None:
        tensors = denorm(tensors)

    tensors = make_grid(tensors, nrow=nrow, padding=2, normalize=True,
                            range=None, scale_each=False, pad_value=0)
    show(tensors)

def Gram(tensor):
    return torch.mm(tensor, tensor.t())

def get_content_style_representations_gatys(content, style):
    # content representations is simply the vectorized features (single scale)
    Cc, Hc, Wc = content.squeeze().shape
    content_repr = content.view(Cc, -1)

    # style will be the Gram matrix of each style activation (multi-scale)
    style_reprs = []
    for _style in style:
        Cs, Hs, Ws = _style.squeeze().shape
        style_repr = _style.view(Cc, -1)
        style_reprs.append(Gram(style_repr))
    
    return content_repr, style_reprs

class FrobeniusNorm(nn.Module):
    """
    simple class handle to use frobenius norm between two matrices in optimisation process
    """
    def __init__(self):
        super(FrobeniusNorm, self).__init__()
    
    def forward(self, x1, x2):
        return torch.linalg.norm((x1-x2), 'fro')