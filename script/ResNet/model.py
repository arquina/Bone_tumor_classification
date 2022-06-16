from torchvision import models
from torch import nn

def classification_model(pretrained, num_classes):

    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
