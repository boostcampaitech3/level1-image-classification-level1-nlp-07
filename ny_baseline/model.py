import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary as summary


# resnet18을 베이스로 만든 모델
# 출력뒤에 18개의 output을 뱉는 fc층만 추가함
# 그 추가한 층은 xavier 초기화
class TunedResnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.fc_to_fc2_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, num_classes)
        self.init_param()
        
    def init_param(self):
        # xavier uniform
        torch.nn.init.xavier_uniform_(self.fc2.weight) 
        stdv = 1. / math.sqrt(self.fc2.weight.size(1))
        self.fc2.bias.data.uniform_(-stdv, stdv)
        
    def forward(self,x):
        x = self.model(x)
        x = F.relu(x)
        x = self.fc_to_fc2_dropout(x)
        return self.fc2(x)
    
    def print_model(self):
        print("네트워크 필요 입력 채널 개수", self.model.conv1.weight.shape[1])
        print("네트워크 출력 채널 개수 (예측 class type 개수)", self.model.fc2.weight.shape[0])
#         print("네트워크 구조", self.model)
        summary(self.model, (3, 512, 384), device='cpu') # (model, input_size)
    

# pretrained Resnet18
class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
    
#     def print_model(self):
#         print("네트워크 필요 입력 채널 개수", self.model.conv1.weight.shape[1])
#         print("네트워크 출력 채널 개수 (예측 class type 개수)", self.fc.weight.shape[0])
# #         print("네트워크 구조", self.model)
#         summary(self.model, (3, 512, 384), device='cpu') # (model, input_size)

    
# if __name__ == "__main__":
#     model = Resnet18(num_classes=18)
#     model.print_model()
    
