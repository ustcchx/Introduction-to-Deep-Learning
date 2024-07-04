import torch.nn as nn
import torch

class BLOCK(nn.Module) :
    
    def __init__(self,inplanes:int, planes:int, stride:int, downsample=None) :
        
        super(BLOCK,self).__init__() #繼承父類
        self.conv1 = nn.Conv2d(inplanes,planes,3,stride,padding=1,bias=False) #子模塊第一層
        self.bn1 = nn.BatchNorm2d(planes) #子模塊第一層標準化
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,planes,3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self,x) :
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
    
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None :
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class ResNet(nn.Module) :
    
    def __init__(self,lays:list,num_classes:int,p: float) :
        super(ResNet,self).__init__()
        self.inplanes = 64
        #self.conv1 = nn.Conv2d(3,self.inplanes,7,stride=2,padding=3)
        self.conv1 = nn.Conv2d(3,self.inplanes,3,stride=1,padding=1)
        #稍微修改一下第一次的捲積操作，便於保留特徵
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1) #後面池化操作進行刪除
        self.lay1 = self._make_lay(64,lays[0],stride=1)
        self.lay2 = self._make_lay(128,lays[1],stride=2)
        self.lay3 = self._make_lay(256,lays[2],stride=2)
        self.lay4 = self._make_lay(512,lays[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_lay(self,planes:int,num_blocks:int,stride:int) -> nn.Sequential :
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        lays = []
        lays.append(BLOCK(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(num_blocks) :
            lays.append(BLOCK(self.inplanes,planes,stride=1))
        return nn.Sequential(*lays)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
         # x = self.maxpool(x)
    
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
    
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
