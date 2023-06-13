import torch.nn as nn
import torch

# preselected set of layer indices that we deem useful for the semantic mask
USEFUL_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class SemanticMask(nn.Module):
    def __init__(self, semantic_net, useful_layers=None):
        super(SemanticMask, self).__init__()

        # if no useful layers are specified, use the default
        self.useful_layers = useful_layers if useful_layers is not None else USEFUL_LAYERS
        self.semantic_net = semantic_net

    def forward(self, x0):
        # run the semantic segmentation network
        semantic_activations = self.semantic_net.forward(x0)

        # zero out the one that we don't want
        nb_channels = semantic_activations.shape[1]
        for i in range(nb_channels):
            if i not in self.useful_layers:
                semantic_activations[:, i] = 0

        # average the channels pixel-wise
        mask = torch.mean(semantic_activations, dim=1)

        return mask
    

class SemanticVprNet(nn.Module):
    def __init__(self, main_model, semantic_net):
        super(SemanticVprNet, self).__init__()

        # save the main model and the semantic segmentation network
        self.main_model = main_model # number of input channels needs to be 4
        self.semantic_net = SemanticMask(semantic_net)

    def forward(self, x0):
        # get the semantic segmentation mask
        mask = self.semantic_net.forward(x0)

        # concatenate the mask to the input
        x0_with_mask = torch.cat((x0, mask.unsqueeze(1)), dim=1)

        # run the main model with the added mask        
        out = self.main_model.forward(x0_with_mask)
        return out