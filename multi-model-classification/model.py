import torchvision.models as models
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self ,num_classes: int = 1000):
        super(MyNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.fc2 = nn.Linear(1000,num_classes)
    def forward(self,x):
        x = self.model(x)
        return self.fc2(x)
    