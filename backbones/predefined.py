import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights, VGG16_Weights

def get_backbone(backbone_name, num_classes, freeze_layers=True):
    if backbone_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Sostituisce il classificatore finale
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Congelare i primi livelli
        if freeze_layers:
            for name, param in model.named_parameters():
                if not name.startswith("layer4") and not name.startswith("fc"):
                    param.requires_grad = False

    elif backbone_name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)

        # Sostituisce il classificatore finale
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

        # Congelare i primi 3 blocchi convoluzionali
        if freeze_layers:
            for i in range(10):  # Blocchi 0-9 corrispondono ai primi 3 blocchi
                for param in model.features[i].parameters():
                    param.requires_grad = False

    else:
        raise ValueError(f"Backbone {backbone_name} non supportata.")
    
    return model
