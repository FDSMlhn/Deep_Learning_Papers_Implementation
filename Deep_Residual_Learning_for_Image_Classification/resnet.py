import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, out_planes, downsample = False):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.downsample = downsample
        self.stride = 1
        if self.downsample:
            self.stride = 2
            self.shortcut_block = nn.Sequential(
                nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, stride=(2,2), bias=False),
                nn.BatchNorm2d(out_planes)
                )

        self.block = nn.Sequential(
            nn.Conv2d(self.in_planes, self.out_planes, kernel_size= 3, stride=self.stride, padding=1,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(self.out_planes, self.out_planes, kernel_size= 3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        
    def forward(self,x):
        shortcut = x
        out = self.block(x)
        if self.downsample:
            shortcut = self.shortcut_block(shortcut)
            out += shortcut
            
        out =  F.relu(out)
        return out
    
class Resnet(nn.Module):
    
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.plane = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.layer1= self._make_layers(block, 16, num_blocks[0], downsample=False)
        self.layer2= self._make_layers(block, 32, num_blocks[1], downsample=True)
        self.layer3= self._make_layers(block, 64, num_blocks[2], downsample=True)
        self.linear= nn.Linear(64*block.expansion, num_classes)
    
    def _make_layers(self, block,in_plane, num_block, downsample):
        layers = []
        for i in range(num_block):
            layers.append(block(self.plane, in_plane, downsample=downsample))
            self.plane= in_plane * block.expansion
            downsample=False
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x=  x.view(x.size(0),-1)
        return self.linear(x)

    
def Resnet20():
    net = Resnet(BasicBlock, [3,3,3], 10)
    return net