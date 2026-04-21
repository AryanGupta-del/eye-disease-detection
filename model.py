import torch
import torch.nn as nn
from torchvision import models

class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EyeDiseaseModel, self).__init__()
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def get_model(num_classes=8):
    model = EyeDiseaseModel(num_classes)
    return model