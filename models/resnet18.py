import torch
import torch.nn as nn

class BasicBlock(nn.Module):
  def __init__(self,in_channels,out_channels,stride=1):
    super().__init__()
    if in_channels!=out_channels or stride!=1:
      self.downsample=nn.Sequential(
          nn.Conv2d(
              in_channels=in_channels,
              out_channels=out_channels,
              kernel_size=1,
              bias=False,
              stride=stride
          ),
          nn.BatchNorm2d(out_channels)
      )
    else:
      self.downsample=None
    self.conv1=nn.Conv2d(
        in_channels=in_channels,
        kernel_size=3,
        padding=1,
        stride=stride,
        bias=False,
        out_channels=out_channels
        )
    self.relu=nn.ReLU()
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2=nn.Conv2d(
         in_channels=out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=False,
        out_channels=out_channels
    )
    self.bn2 = nn.BatchNorm2d(out_channels)
  def forward(self,x): #conv-bn-relu-conv-bn-res-relu
      identity=x
      if self.downsample:
        identity=self.downsample(x)
      out=self.conv1(x)
      out=self.bn1(out)
      out=self.relu(out)
      out=self.conv2(out)
      out=self.bn2(out)
      out+=identity
      out=self.relu(out)
      return out

class ResNet18(nn.Module):
  def __init__(self,num_classes):
    super().__init__()
    self.conv1=nn.Conv2d(3,64,stride=2,padding=3,kernel_size=7,bias=False)
    self.bn1=nn.BatchNorm2d(64)
    self.relu=nn.ReLU()
    self.maxpool=nn.MaxPool2d(3,stride=2,padding=1)

    self.layer1=nn.Sequential(
        BasicBlock(64,64,stride=1),
        BasicBlock(64,64,stride=1),
    )
    self.layer2=nn.Sequential(
        BasicBlock(64,64,stride=2),
        BasicBlock(64,128,stride=1),
    )
    self.layer3=nn.Sequential(
        BasicBlock(128,128,stride=2),
        BasicBlock(128,256,stride=1),
    )
    self.layer4=nn.Sequential(
        BasicBlock(256,256,stride=2),
        BasicBlock(256,512,stride=1),
    )


    self.avgpool=nn.AdaptiveAvgPool2d((1,1))

    self.fc=nn.Linear(512,num_classes)
  def forward(self,x):
    out=self.conv1(x)
    out=self.bn1(out)
    out=self.relu(out)
    out=self.maxpool(out)

    out=self.layer1(out)
    out=self.layer2(out)
    out=self.layer3(out)
    out=self.layer4(out)
    out=self.avgpool(out)
    out=torch.flatten(out,1)
    out=self.fc(out)
    return out
