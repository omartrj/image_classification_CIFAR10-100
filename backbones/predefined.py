import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights, VGG16_Weights


def get_backbone(backbone_name, num_classes, freeze_layers_up_to_block=2):
    if backbone_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Congela solo i primi freeze_layers_up_to_block blocchi
        layers_to_freeze = list(model.children())[:freeze_layers_up_to_block]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        # Sostituisce il classificatore finale
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    else:
        raise ValueError(f"Backbone {backbone_name} non supportata.")
    
    return model
