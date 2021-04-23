import torch
import torch.nn as nn

class modelWithActivations(nn.Module):
    """
    simple class handle to return forward hooks of a torch.nn DL model
    """
    def __init__(self, model):
        super(modelWithActivations, self).__init__()

        assert isinstance(model, nn.Module), "input parameter ``model`` should be an <nn.Module>"\
                                             " instance, got %s"%type(model)
        self.features = model.features

    def forward(self, x):
        return self.features[:](x)

    def forward_with_activations(self, x):
        activations = []
        for _, module in self.features._modules.items():
            x = module(x)
            activations.append(x)
        return activations
    
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def switch_to_avg_pooling(self):
        # replace the MaxPool with the AvgPool layers 
        # (original Gatys et al. paper showed improved results)
        for name, child in self.features.named_children():
            if isinstance(child, nn.MaxPool2d):
                self.features[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)