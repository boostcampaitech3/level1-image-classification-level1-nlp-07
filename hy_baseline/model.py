import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import torchvision.models._utils as _utils

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

    
class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-5])
    def forward(self,x):
        return self.model(x)
    
class SubNet(nn.Module):
    def __init__(self,num_class=2):
        super(SubNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[-5:-1])
        self.fc = nn.Linear(in_features=2048, out_features=num_class, bias=True)
    def forward(self,x):
        x=self.model(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x)
        return x        
        
class Multi_ModelClassification(nn.Module):
    def __init__(self, phase = 'train'):
        super(Multi_ModelClassification,self).__init__()
        self.head = ClassificationHead()
        
        self.sub_age = SubNet(3)
        self.sub_gender = SubNet(2)
        self.sub_mask = SubNet(3)

    def forward(self,inputs):
        out = self.head(inputs) #pretrained model

        # SSH
        feature_age = self.sub_age(out)
        feature_gender = self.sub_gender(out)
        feature_mask = self.sub_mask(out)
        
            
        return [feature_age, feature_gender, feature_mask]

    
# class ClassificationHead(nn.Module):
#     def __init__(self):
#         super(ClassificationHead, self).__init__()
#         self.model = models.resnet34(pretrained=True)
#         self.model = nn.Sequential(*list(self.model.children())[:-4])
#     def forward(self,x):
#         return self.model(x)
    
# class SubNet(nn.Module):
#     def __init__(self,num_class=2):
#         super(SubNet, self).__init__()
#         self.model = models.resnet34(pretrained=True)
#         self.model = nn.Sequential(*list(self.model.children())[-4:-1])
#         self.fc = nn.Linear(in_features=512, out_features=num_class, bias=True)
#     def forward(self,x):
#         x=self.model(x)
#         x=x.view(x.size(0), -1)
#         x=self.fc(x)
#         return x        
        
# class Multi_ModelClassification(nn.Module):
#     def __init__(self, phase = 'train'):
#         super(Multi_ModelClassification,self).__init__()
#         self.head = ClassificationHead()
        
#         self.sub_age = SubNet(3)
#         self.sub_gender = SubNet(2)
#         self.sub_mask = SubNet(3)

#     def forward(self,inputs):
#         out = self.head(inputs) #pretrained model

#         # SSH
#         feature_age = self.sub_age(out)
#         feature_gender = self.sub_gender(out)
#         feature_mask = self.sub_mask(out)
        
            
#         return [feature_age, feature_gender, feature_mask]