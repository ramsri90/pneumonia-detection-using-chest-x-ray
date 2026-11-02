import torch.nn as nn
import torchvision.models as models
def build_model():
    model=models.resnet18(pretrained=True)
    model.fc=nn.Linear(model.fc.in_features,2)
    return model