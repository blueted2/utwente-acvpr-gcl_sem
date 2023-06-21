import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


# # preselected set of layer indices that we deem useful for the semantic mask
# USEFUL_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# class SemanticMask(nn.Module):
#     def __init__(self, semantic_net, useful_layers=None):
#         super(SemanticMask, self).__init__()

#         semantic_net.eval()

#         # if no useful layers are specified, use the default
#         self.useful_layers = (
#             useful_layers if useful_layers is not None else USEFUL_LAYERS
#         )
#         self.semantic_net = semantic_net

#         # freeze the semantic segmentation network
#         for param in self.semantic_net.parameters():
#             param.requires_grad = False

#     def forward(self, x0):
#         # run the semantic segmentation network
#         semantic_activations = self.semantic_net.forward(x0)

#         # set non-useful layers to zero
#         for i in range(semantic_activations.shape[1]):
#             if i not in self.useful_layers:
#                 semantic_activations[:, i, :, :] = 0

#         # average the useful layers
#         mean_activations = torch.mean(semantic_activations, dim=1, keepdim=True)

#         return mean_activations


# class LearnedWeightedSemanticMask(nn.Module):
#     def __init__(self, semantic_net):
#         super(LearnedWeightedSemanticMask, self).__init__()

#         # if no useful layers are specified, use the default
#         self.semantic_net = semantic_net

#         # freeze the semantic segmentation network
#         for param in self.semantic_net.parameters():
#             param.requires_grad = False

#         self.conv = nn.Conv2d(19, 1, kernel_size=1, stride=1, padding=0, bias=False)

#         self.cache = {}

#     def forward(self, x0):
#         # check if we have already computed the semantic segmentation for this image
#         if x0 in self.cache:
#             semantic_activations = self.cache[x0].to(x0.device)
#         else:
#             # run the semantic segmentation network
#             semantic_activations = self.semantic_net.forward(x0)
#             self.cache[x0] = semantic_activations.detach().cpu()

#         # apply the convolution
#         mask = self.conv(semantic_activations)

#         del semantic_activations
#         return mask


# # a wrapper class that adds some kind of augmentation to the input, beit a semantic mask or all 19 semantic channels
# class AugmentedVprBackbone(nn.Module):
#     def __init__(self, main_model, augmentation_net):
#         super(AugmentedVprBackbone, self).__init__()

#         # save the main model and the semantic segmentation network
#         self.augmentation_net = augmentation_net
#         self.main_model = main_model

#     def forward(self, x0):
#         # get the semantic segmentation mask
#         augmentation = self.augmentation_net.forward(x0)

#         # concatenate the mask to the input
#         x0_with_mask = torch.cat((x0, augmentation), dim=1)

#         # run the main model with the added mask
#         out = self.main_model.forward(x0_with_mask)
#         return out


class SingleNet(nn.Module):
    def __init__(self, backbone):
        super(SingleNet, self).__init__()
        self.backbone = backbone

    def forward(self, x0):
        out = self.backbone.forward(x0)
        return nn.functional.normalize(out)


class SiameseNet(nn.Module):
    def __init__(self, backbone):
        super(SiameseNet, self).__init__()
        self.single_net = SingleNet(backbone)

    def forward(self, x0, x1):
        out0 = self.single_net.forward(x0)
        out1 = self.single_net.forward(x1)
        return out0, out1


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, out0, out1, label):
        gt = label
        D = self.distance(out0, out1).float().squeeze()
        loss = gt * 0.5 * torch.pow(D, 2) + (1 - gt) * 0.5 * torch.pow(
            torch.clamp(self.margin - D, min=0.0), 2
        )
        return loss.mean()
