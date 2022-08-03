from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self,prob : float = 0):
        super(TestModel,self).__init__()

        self.conv1 = nn.Conv2d(1,8,(3,3),padding=(1,1)) #28
        self.conv2 = nn.Conv2d(8,4,(3,3),padding=(1,1)) #14
        self.conv3 = nn.Conv2d(4,1,(3,3),padding=(1,1)) #7
        self.relu = nn.LeakyReLU()
        self.pooling = nn.AvgPool2d((2,2))

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(7*7,16)
        self.fc2 = nn.Linear(16,10)

        self.bn2d_1 = nn.BatchNorm2d(4)
        self.bn2d_2 = nn.BatchNorm2d(1)

        self.bn1d_1 = nn.BatchNorm1d(16)
        self.bn1d_2 = nn.BatchNorm1d(10)

        if prob != 0:
            self.dropout1 = nn.Dropout(p=prob)
            self.dropout2 = nn.Dropout(p=prob)
            self.dropout3 = nn.Dropout(p=prob)

    def forward(self, x : Tensor):

        x = self.relu(self.dropout1(self.conv1(x)))
        x = self.bn2d_1(self.pooling(self.relu(self.dropout2(self.conv2(x)))))
        x = self.bn2d_2(self.pooling(self.relu(self.dropout3(self.conv3(x)))))

        x = self.flatten(x)

        x = self.bn1d_1(self.relu(self.fc1(x)))
        x = self.bn1d_2(self.relu(self.fc2(x)))

        return F.log_softmax(x)