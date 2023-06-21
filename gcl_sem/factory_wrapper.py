from gcl_sem.libs.generalized_contrastive_loss.src import factory as gcl_factory
from gcl_sem.libs.generalized_contrastive_loss.src.networks import SiameseNet, BaseNet
from gcl_sem.libs.deeplabv3plus_pytorch.network import modeling
from .networks import SemanticVprNet

import torch


# wrapper for gcl_factory.get_backbone which allows changing the number of input channels
def create_vpr_backbone(name, in_channels=3):
    backbone, output_dim = gcl_factory.get_backbone(name)
    backbone[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return backbone, output_dim

def create_vpr_model(name, pool, last_layer=None, norm=None, p_gem=3, mode="siamese", in_channels=3):
    backbone, output_dim = create_vpr_backbone(name, in_channels)
    layers = len(list(backbone.children()))

    if last_layer is None:
        last_layer = layers
    elif "densenet" in name:
        last_layer=last_layer*2
    elif "vgg" in name:
        last_layer=last_layer*8-2
    aux = 0
    for c in backbone.children():

        if aux < layers - last_layer:
            # print(aux, c._get_name(), "IS FROZEN")
            for p in c.parameters():
                p.requires_grad = False
        else:
            # print(aux, c._get_name(), "IS TRAINED")
            pass
        aux += 1
    if mode=="siamese":
        return SiameseNet(backbone, pool, norm=norm, p=p_gem)
    else:
        return BaseNet(backbone, pool, norm=norm, p=p_gem)

# TODO take parameters
def create_semantic_net():

    # TODO absolute path bad plz fix
    weights = "/home/s3155900/gregory/generalized_contrastive_loss/deep_lab_v3/best_deeplabv3plus_resnet101_cityscapes_os16.pth"

    # instantiate a semantic segmentation model and load with pretrained weights
    semantic_net = modeling.deeplabv3plus_resnet101(num_classes=19)
    semantic_net.load_state_dict( torch.load( weights, map_location=torch.device('cpu') )['model_state']  )
    
    return semantic_net
    

# TODO update with 
def create_semantic_vpr_model(backbone_name='resnet50', in_channels=3):
    vpr_model = create_vpr_model(backbone_name, pool="GeM", last_layer=2, norm="L2", p_gem=3, mode="single", in_channels=in_channels)
    sem_net = create_semantic_net()

    return SemanticVprNet(vpr_model, sem_net)

