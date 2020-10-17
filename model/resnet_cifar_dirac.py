import math, torch, pdb
import torch.nn.functional as F
import torch.nn as nn
from .diraconv import DiracConv2d
inplace = False
affine = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3_dirac(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return DiracConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1_dirac(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return DiracConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine)
        self.conv2 = conv3x3_dirac(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine)
        self.conv3 = conv1x1_dirac(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        x   = self.relu(x)
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class Bottleneckwoskip(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneckwoskip, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.conv2 = conv3x3_dirac(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.conv3 = conv1x1_dirac(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.stride = stride
        self.downsample = downsample
        self.connector1 = nn.Conv2d(planes, inplanes, kernel_size=1)
        self.connector2 = nn.Conv2d(planes * self.expansion, planes, kernel_size=1)

    def forward(self, x, **kwargs):
        out = self.relu(x)
        
        #id1 = self.conv1.forward_skip(out.detach())
        #(self.connector1(id1) - out.detach()).mean().backward()
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        return out
    

    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3_dirac(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv2 = conv3x3_dirac(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x, **kwargs):
        x   = self.relu(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
    
class BasicBlockwoskip(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockwoskip, self).__init__()

        self.conv1 = conv3x3_dirac(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv2 = conv3x3_dirac(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        x   = self.relu(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        return out   
          
class ResNet(nn.Module):
    def __init__(self, block, blockwoskip, layers_blocks, blockchoice, num_classes=100, mul=1.0):
        super(ResNet, self).__init__()
        self.mul = mul
        self.inplanes = int(64 * self.mul)
        self.layers = nn.ModuleList()
        self.blockchoice = blockchoice
        self.layer_blocks= layers_blocks
        self.block = block
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64 * self.mul), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * self.mul), affine=affine),
            nn.ReLU(inplace=True))

        self.layers.append(self._make_layer(block, blockwoskip, self.blockchoice[0], int(64*self.mul),  layers_blocks[0]          ))
        self.layers.append(self._make_layer(block, blockwoskip, self.blockchoice[1], int(128*self.mul), layers_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, blockwoskip, self.blockchoice[2], int(256*self.mul), layers_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, blockwoskip, self.blockchoice[3], int(512*self.mul), layers_blocks[3], stride=2))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512*self.mul) * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        zero_init_residual = False
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, blockwoskip, blockchoice, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, affine=affine),
            )
        layers = []
        thisblock = block if blockchoice[0] == 1 else blockwoskip
        layers.append(thisblock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            thisblock = block if blockchoice[i] == 1 else blockwoskip
            layers.append(thisblock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                x = self.layers[i][j](x, **kwargs)
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward_towindow(self, x, **kwargs):
        dis_point= kwargs.get("dis_point")
        dis_fea  = []
        layer_index = 0
        
        x = self.conv1(x)
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                x = self.layers[i][j](x)
                if layer_index in dis_point:
                    dis_fea.append(x)
                layer_index += 1 
            if i == 1:
                x = x.detach()
                    
        dis_fea.append(x)
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, dis_fea
    
    def forward_bt(self, x, **kwargs):
        x = self.conv1(x)
        return x
    
    def forward_bl(self, x, **kwargs):
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward_to(self, x, **kwargs):
        dis_point = kwargs.get("dis_point")
        dis_fea  = []
        layer_index = 0
        x = self.conv1(x)
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                x = self.layers[i][j](x)
                if layer_index in dis_point:
                    dis_fea.append(x)
                layer_index += 1 
        dis_fea.append(x)
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, dis_fea
                
    def forward_from(self, x, **kwargs):
        se_index = kwargs.get("se_index")
        layer_index = 0
        dis_fea = []
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                if layer_index < se_index:
                    layer_index += 1
                else:
                    x = self.layers[i][j](x)
        dis_fea.append(x)      
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, dis_fea
    
    def get_bn_before_relu(self, dis_point):
        
        layer_index = 0
        bn = []
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                if layer_index in dis_point:
                    if isinstance(self.layers[i][j], BasicBlock):
                        bn.append(self.layers[i][j].bn2)
                    else:
                        bn.append(self.layers[i][j].bn3)
                layer_index += 1
        return bn
                
    def get_layer_blocks(self):
        return self.layer_blocks
    
    def get_blockchoice(self):
        return self.blockchoice
    
    def get_channel_num(self, dis_point):
        layer_index = 0
        channels = []
        channel = [[64 * 4]*3, [128* 4]*4, [256* 4]*6, [512* 4]*3]
        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                if layer_index in dis_point:
                    channels.append(channel[i][j])
                layer_index += 1
        return channels
    
    def get_base_channel(self, dis_point):
        num = len(dis_point) + 1
        if self.block == Bottleneck:
            return [2048]*num
        elif self.block == BasicBlock:
            return [512]*num
        
        '''
                channels = None
        if self.block == Bottleneck:
            channels = [[64 * 4]*3, [128* 4]*4, [256* 4]*6, [512* 4]*3]
        elif self.block == BasicBlock:
            channels = [[64]*3, [128]*4, [256]*6, [512]*3]
        channels = [c for cg in channels for c in cg ]
        res = []
        for point in dis_point:
            res.append(channels[point])
        
        if self.block == Bottleneck:
            res.append(2048)
        elif self.block == BasicBlock:
            res.append(512)
        return res
        
        '''

    
#block, blockwoskip, layers, blockchoice, num_classes=1000
def resnet50(blockchoice, num_classes=100, mul=1.0):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(Bottleneck, Bottleneckwoskip, [3, 4, 6, 3], blockchoice, num_classes, mul=mul)

def resnet34(blockchoice, num_classes=100, mul=1.0):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(BasicBlock, BasicBlockwoskip, [3, 4, 6, 3], blockchoice, num_classes, mul=mul)

def resnet18(blockchoice, num_classes=100, mul=1.0):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(BasicBlock, BasicBlockwoskip, [2, 2, 2, 2], blockchoice, num_classes, mul=mul)

if __name__ == "__main__":
    blockchoice = [[0]*3, [1]*4, [1]*6, [1]*3]
    model = resnet50(blockchoice)
    print(model)