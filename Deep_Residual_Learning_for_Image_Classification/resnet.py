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
        self.shortcut_block= nn.Sequential()
        if self.downsample:
            self.stride = 2
            self.shortcut_block = nn.Sequential(
                nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, stride=self.stride, bias=False),
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
        out = self.block(x)
        out += self.shortcut_block(x)
        out =  F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, out_planes, downsample = False):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.downsample = downsample
        self.stride = 1
        self.shortcut_block= nn.Sequential()
        if self.downsample or self.in_planes != self.out_planes * expansion:
            self.stride = 2
            self.shortcut_block = nn.Sequential(
                nn.Conv2d(self.in_planes, self.out_planes*expansion , kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_planes*expansion)
                )

        self.block = nn.Sequential(
            nn.Conv2d(self.in_planes, self.out_planes, kernel_size= 1, stride = self.stride, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(self.out_planes, self.out_planes, kernel_size= 3, stride=1, padding = 1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(self.out_planes, self.out_planes * expansion, kernel_size= 1, stride = 1, bias=False),
            nn.BatchNorm2d(out_planes * expansion),
        )
        
    def forward(self,x):
        out = self.block(x)
        out += self.shortcut_block(x)
        out =  F.relu(out)
        return out    
    
    

class Resnet(nn.Module):
    
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.plane = 64
        self.conv = nn.Conv2d(3, self.plane, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn = nn.BatchNorm2d(self.plane)
        self.layer1= self._make_layers(block, 64, num_blocks[0], downsample=False)
        self.layer2= self._make_layers(block, 128, num_blocks[1], downsample=True)
        self.layer3= self._make_layers(block, 256, num_blocks[2], downsample=True)
        self.layer4= self._make_layers(block, 512, num_blocks[3], downsample=True)
        
        self.linear= nn.Linear(512*block.expansion, num_classes)
    
    def _make_layers(self, block,in_plane, num_block, downsample):
        layers = []
        for i in range(num_block):
            layers.append(block(self.plane, in_plane, downsample=downsample))
            self.plane= in_plane * block.expansion
            downsample=False
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.bn(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=4)
        out = out.view(out.size(0),-1)
        return self.linear(out)

    
def Resnet18():
    net = Resnet(BasicBlock, [2,2,2,2])
    return net

def Resnet50()
    net = Resnet(Bottleneck, [3,4,6,3])
    return net