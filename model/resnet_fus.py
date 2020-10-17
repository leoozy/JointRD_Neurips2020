import math, torch, pdb
import torch.nn.functional as F
import torch.nn as nn
inplace = False


blockchoice_cfg = {
    "A":[[1]*3,   [1]*4,   [1,1,1,1,1,1], [1,1,1]],
    "B":[[1]*3,   [1]*4,   [1,1,1,1,1,1], [1,0,1]],
    "C":[[1]*3,   [1]*4,   [1,0,1,0,1],   [1,0,1]],
    "D":[[1]*3,   [1,0,1], [1,0,1,0,1],   [1,0]],
    "E":[[1,1,1], [1,0,1], [1,1,0,1],     [1,1]],
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, exp_rate=4):
        super(Bottleneck, self).__init__()
        self.expansion = exp_rate
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=inplace)
        self.downsample = downsample
        self.stride = stride
        self.out_channel = planes * exp_rate
        
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, exp_rate=4):
        super(Bottleneckwoskip, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=inplace)
        self.stride = stride
        self.downsample = downsample
        self.out_channel = planes * 4

    def forward(self, x, **kwargs):
        x   = self.relu(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out
    
class NormalCov(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, downsample=None, exp_rate=4):
        super(NormalCov, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv = conv3x3(inplanes, planes)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=inplace)
        self.out_channel = planes


    def forward(self, x, **kwargs):
        
        x   = self.relu(x)
        out = self.conv(x)
        out = self.bn(out)

        return out
        
class ResNet(nn.Module):

    def __init__(self, block, NormalCov, layers_blocks, blockchoice, num_classes=1000, mul=1.0):
        super(ResNet, self).__init__()
        self.inplanes    = int(mul * 64)
        self.conv1       = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1         = nn.BatchNorm2d(self.inplanes)
        self.relu        = nn.ReLU(inplace=inplace)
        self.maxpool     = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers      = nn.ModuleList()
        self.blockchoice = blockchoice
        layers_blocks = []
        for i in blockchoice:
            layers_blocks.append(len(i))
        self.layer_blocks= layers_blocks
        self.layers.append(self._make_layer(block, NormalCov, self.blockchoice[0], int(mul * 64),  layers_blocks[0]          ))
        self.layers.append(self._make_layer(block, NormalCov, self.blockchoice[1], int(mul * 128), layers_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, NormalCov, self.blockchoice[2], int(mul * 256), layers_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, NormalCov, self.blockchoice[3], int(mul * 512), layers_blocks[3], stride=2))
        self.se_index = None
        self.batch_pro = None
        self.dis_point = None
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(self.inplanes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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
        exp_rate = 1 if (0 <= (len(blockchoice)-2) and blockchoice[0+1]==0) else 4
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * exp_rate, stride),
                nn.BatchNorm2d(planes * exp_rate),
            )
        layers = []
        thisblock = block if blockchoice[0] == 1 else Bottleneckwoskip
        layers.append(thisblock(self.inplanes, planes, stride, downsample, exp_rate=exp_rate))
        self.inplanes = layers[-1].out_channel
        for i in range(1, blocks):
            thisblock = block if blockchoice[i] == 1 else NormalCov
            exp_rate = 1 if (i <= (len(blockchoice)-2) and blockchoice[i+1]==0) else 4
            if self.inplanes != planes * exp_rate:
                downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * exp_rate, stride=1),
                nn.BatchNorm2d(planes * exp_rate),
                )
            else:
                downsample = None
            layers.append(thisblock(self.inplanes, planes, downsample=downsample, exp_rate=exp_rate))
            self.inplanes = layers[-1].out_channel
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                x = self.layers[i][j](x, **kwargs)
        x = F.relu(x, inplace=inplace)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward_towithwin(self, x):
        dis_point= self.dis_point
        dis_fea  = []
        layer_index = 0
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layer_blocks)):
            for j in range(self.layer_blocks[i]):
                x = self.layers[i][j](x)
                if layer_index in self.dis_point:
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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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
                    print(i,j)
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
    
    def get_base_channel(self):
        return 16
    
#block, blockwoskip, layers, blockchoice, num_classes=1000
def resnet50(blockchoice_id, num_classes=1000, mul=1.0):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(Bottleneck, NormalCov, [3, 4, 6, 3], blockchoice_cfg[blockchoice_id], num_classes, mul=mul)



if __name__ == "__main__":
    import time
    model = resnet50("C", num_classes=1000, mul=1.0).cuda()
    print(model)
    time_list = []
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic= False
    for i in range(100):
        if i >=1 :
            inp = torch.randn(10, 3, 224, 224).cuda()
            torch.cuda.synchronize()
            st = time.time()
            out = model(inp)
            torch.cuda.synchronize()
            this_time = time.time()-st
            time_list.append(this_time)
    print(time_list)
    print(sum(time_list[2:])/len(time_list[2:]))