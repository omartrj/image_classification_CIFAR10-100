import torch.nn as nn
from backbones.predefined import get_backbone
from backbones.custom import CustomBackbone


class Net(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super(Net, self).__init__()
        if backbone_name == "custom":
            self.backbone = CustomBackbone(num_classes=num_classes, use_stride=False)
        elif backbone_name == "resnet50":
            self.backbone = get_backbone(
                backbone_name=backbone_name,
                num_classes=num_classes,
                freeze_layers_up_to_block=3,
            )
        else:
            raise ValueError(
                f"Backbone '{backbone_name}' non supportata. Usa 'resnet50' o 'custom'."
            )

    def forward(self, x):
        return self.backbone(x)
