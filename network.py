#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import partial
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from modules.modulated_deform_conv import ModulatedDeformConvPack

from bn_lib.nn.modules import SynchronizedBatchNorm2d
import settings

norm_layer = partial(SynchronizedBatchNorm2d, momentum=settings.BN_MOM)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        # self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation,
        #                        bias=False)
        self.conv2 = ModulatedDeformConvPack(planes, planes, kernel_size=(3, 3), stride=stride,
              padding=dilation, dilation=dilation, deformable_groups=2).cuda()
        self.bn2 = norm_layer(planes)
        #此处4应该替代为self.expansion，是bottleneck实现方式的压缩再放大形式
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, stride=8):
        self.inplanes = 128
        super().__init__()
        # 结构图中第一个卷积层，即7*7卷积核MAX_poolling层的构造，这里是Resnet的改版，使用的是
        # 三个3*3卷积核替代了该7*7卷积核，由于pytorch中resnet的实现形式还是7*7的，因此调用效果并不好
        # 所以文章作者选择了自己重写Resnet的网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 这里实现了将通道数从128变成64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 此处为分别构造4个stage的过程，resnet50对应的layers为3 4 6 3
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if stride == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(
                    block, 512, layers[3], stride=1, dilation=2, grids=[1,2,4])
        elif stride == 8:
            self.layer3 = self._make_layer(
                    block, 256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(
                    block, 512, layers[3], stride=1, dilation=4, grids=[1,2,4])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    # 这个是作者自己定义的一个Resnet的构造方式，与传统的Resnet不同，其增加了一个grids参数
    # 这个参数是为了决定每个层中的带孔卷积的孔径大小的
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    grids=None):
        downsample = None
        # 这里的下采样其实就是一个防止报错和修正通道的过程，当通道数不OK时，或者是由于stride变化
        # 造成输出的大小减半时，这时候通过下采样对residual部分进行配平
        # 这里采用[1 2 4] 这样的grid 感觉是为了像deeplab那样融合金字塔结构，获取不一样的感受野
        # 分割的时候这样的效果更加的好，作者提到过这么用的原因。
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        if grids is None:
            # list * int 意思是将数组重复 int 次并依次连接形成一个新数组
            grids = [1] * blocks

        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample,
                                previous_dilation=dilation))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample,
                                previous_dilation=dilation))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=dilation*grids[i],
                                previous_dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(n_layers, stride):
    layers = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[n_layers]
    pretrained_path = {
        50: './models/resnet50-ebb6acbb.pth',
        101: './models/resnet101-2a57e44d.pth',
        152: './models/resnet152-0d43d698.pth',
    }[n_layers]

    net = ResNet(Bottleneck, layers=layers, stride=stride)
    state_dict = torch.load(pretrained_path)
    net.load_state_dict(state_dict, strict=False)

    return net


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        # stage_num 是指定的迭代次数
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        # 取一个正态分布，N(0,2/n) 其中n为当前层中神经元的数量（全链接层）
        mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        # 因此需要进行标准化，BN和LN都是很好的选择，但是这些方法会改变更新偏差值u^k的方向，
        # 这会改变其属性和语义，为了保持方向不变，作者使用了L2Norm，通过向量长度去分配u|k^t,
        # 最终形成K维超球面
        mu = self._l2norm(mu, dim=1)
        # 我们知道，pytorch一般情况下，是将网络中的参数保存成OrderedDict形式的。这里额参数其实包括2种。
        # 一种是模型中各种 module含的参数，即nn.Parameter，
        # 我们当然可以在网络中定义其他的nn.Parameter参数。；
        # 另外一种是buffer。前者每次optim.step会得到更新，而不会更新后者。
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 求参数量以及kaiming初始化整个weight
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        # 这里没有使用激活层是为了将输出映射至[-∞,+∞]而非[0,＋无穷]
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                # permute 的作用是改变tensor维度的排序
                x_t = x.permute(0, 2, 1)    # b * n * c
                # batch matrix multiply 批量矩阵乘法而已哦啦啦
                z = torch.bmm(x_t, mu)      # b * n * k
                # 论文中所说的那一步，选核函数为exp(aTb)后EM算法的E步与softmax是等价的
                # 而这个核函数计算相似度的过程其实就是non-local中计算不同点之间的关系的一种思想的转换而已
                z = F.softmax(z, dim=2)     # b * n * k
                # 接下来两步就是EM 算法中的M步，迭代更新mu的过程，最后对mu进行L2norm，这个是通过GMM推导中的mu得出的公式
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)       # b * c * k
                # BN和LN会改变每个mu_k的方向，这样对最终结果并不好，因此作者使用了L2NORM
                # 其实作者也是对大多数归一化函数进行了测试，最后发现L2norm效果最好并做出了一些分析
                # 而欧几里得距离（L2norm）不会改变这些方向，通过每个u_k取除以所有u的二范数，是一个标量的方式进行归一化
                mu = self._l2norm(mu, dim=1)

        # 这个过程是不是有点像SVD的过程呢？？？
        z_t = z.permute(0, 2, 1)            # b * k * n
        x = mu.matmul(z_t)                  # b * c * n
        x = x.view(b, c, h, w)              # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        # 通过1*1卷积将维度变回来，作为residual加在原来的X上。
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        # 求指定维度上的范数torch.norm(input, p, dim, out=None,keepdim=False) → Tensor
        # 返回输入张量给定维dim 上每行的p 范数。小的数字是为了防止除以0的错误
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class EMANet(nn.Module):
    ''' Implementation of EMANet (ICCV 2019 Oral).'''
    def __init__(self, n_classes, n_layers):
        super().__init__()
        backbone = resnet(n_layers, settings.STRIDE)
        self.extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4)

        self.fc0 = ConvBNReLU(2048, 512, 3, 1, 1, 1)
        # self.emau = EMAU(512, 64, settings.STAGE_NUM)
        self.fc1 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, 1),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(256, n_classes, 1)

        # Put the criterion inside the model to make GPU load balanced
        self.crit = CrossEntropyLoss2d(ignore_index=settings.IGNORE_LABEL, 
                                       reduction='none')

    def forward(self, img, lbl=None, size=None):
        # 先通过一个resnet 提取出特征x，然后CONV BN RELU一套组合拳，输出网络激活值（该网络就是resnet50或者其他）
        x = self.extractor(img)
        x = self.fc0(x)
        # 通过EMA框架对resnet的激活值进行再处理，这里是用的显著性的方法，通过降维显著性使用的方式对原方法进行压缩
        # x, mu = self.emau(x)
        x = self.fc1(x)
        x = self.fc2(x)

        if size is None:
            size = img.size()[-2:]
        pred = F.interpolate(x, size=size, mode='bilinear', align_corners=True)

        if self.training and lbl is not None:
            loss = self.crit(pred, lbl)
            # return loss, mu
            return loss
        else:
            return pred


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        # 交叉熵CrossEntropyLoss()=log_softmax() + NLLLoss() ，而这个NLLLoss最适合最后已经有softmax的网络
        # nn.BCELoss()是二分类时候用的交叉熵函数，此时sigmoid等同于softmax
        # 即log_softmax计算出的是当目标属于每个类的时候的交叉熵损失，是一个矩阵[]，
        # NLLLoss是根据这个矩阵再乘以设置类别权重计算出最后结果的过程
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, 
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)


def test_net():
    model = EMANet(n_classes=21, n_layers=50)
    model.eval()
    print(list(model.named_children()))
    image = torch.randn(1, 3, 513, 513)
    label = torch.zeros(1, 513, 513).long()
    pred = model(image, label)
    print(pred.size())


if __name__ == '__main__':
    test_net()
