import torch.nn as nn
import torch
from torch import autograd
import math

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=False ):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)

            self.bilinear=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        #fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp

        return fms_att

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,groups=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, groups=4,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(), )
    # cheap operation
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),)

    def forward(self, x):
        x1 = self.primary_conv(x)  
        x2 = self.cheap_operation(x1) 
        out = torch.cat([x1,x2], dim=1) 
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, out_chs, n=1,dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=1):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, in_chs*n, relu=True)

        self.conv_dw = nn.Conv2d(in_chs*n, in_chs*n, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=in_chs*n, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_chs*n)


        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(in_chs*n, in_chs*n, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=in_chs*n, bias=False)
            self.bn_dw = nn.BatchNorm2d(in_chs*n)

        # Squeeze-and-excitation
        if has_se:
            self.se = SEModule(in_chs*n, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(in_chs*n, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential( )

        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs,out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x
        # print(x.shape)

        # 1st ghost bottleneck
        x = self.ghost1(x)
        # print("ghost1",x.shape)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        x = self.conv_dw(x)
        x = self.bn_dw(x)


        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        # print('ghost2:',x.shape)

        x += self.shortcut(residual)
        return x

class Convghost(nn.Module):
    """
    (convolution => [BN] => ReLU) * ghost
    """
    def __init__(self, in_channels, out_channels):
        super(Convghost, self).__init__()
        self.Convghost = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # GhostModule(out_channels, out_channels)
            # GhostBottleneck(in_channels, out_channels),
            GhostBottleneck(out_channels, out_channels),
            GhostBottleneck(out_channels, out_channels),
        )

    def forward(self, x):
        return self.Convghost(x)

class Conv11(nn.Module):
    """
    (convolution => [BN] => ReLU) * 1 1
    """
    def __init__(self, in_channels, out_channels):
        super(Conv11, self).__init__()
        self.Convghost = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=8,padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.Convghost(x)



class twoGhostBottleneck(nn.Module):
    """
    GhostBottleneck*2
    """
    def __init__(self, in_channels, out_channels):
        super(twoGhostBottleneck, self).__init__()
        self.Convghost = nn.Sequential(

            GhostBottleneck(in_channels, out_channels),
            GhostBottleneck(out_channels, out_channels),
        )

    def forward(self, x):
        return self.Convghost(x)


class Doubleghost(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            GhostModule(in_channels, out_channels),
            GhostModule(out_channels, out_channels),
            # GhostModule(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1,0, 1,1,False)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1,0, 1,1,False)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0, 1,1,False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0,1,1, False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c, -1)
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x 
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


class Convp(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(Convp, self).__init__()
        self.Convg = nn.Sequential(
            nn.Conv2d(in_channels, 10*out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(10*out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(10*out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.Convg(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEModule(nn.Module):
    def __init__(self, channels, se_ratio=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // se_ratio, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        # self.relu =Act()
        self.fc2 = nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class mynet(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(myChannelUnet, self).__init__()
        filter = [64,128,256,512,512]
        self.conv1 = DoubleConv(img_ch, filter[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = Convghost(filter[0], filter[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = Convghost(filter[1], filter[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = GhostBottleneck(filter[2], filter[3])
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = GhostBottleneck(filter[3], filter[4])

        self.local = NonLocalBlock(filter[4])
        self.res2 = Res2NetModule(filter[4], filter[4])
        self.rfb = BasicRFB(filter[4], filter[4])
        self.dropout_layer = nn.Dropout()

        self.up6 = nn.ConvTranspose2d(filter[4], filter[3], 2, stride=2,groups=8)
        self.conv6 = GhostModule(filter[3]*3,filter[3])
        # self.conv6 = Conv11 (filter[3]*3,filter[3])
        self.up7 = nn.ConvTranspose2d(filter[3], filter[2], 2, stride=2,groups=8)
        self.conv7 = GhostModule(filter[2]*3, filter[2])
        # self.conv7 = Conv11 (filter[2]*3, filter[2])
        self.up8 = nn.ConvTranspose2d(filter[2], filter[1], 2, stride=2,groups=8)
        self.conv8 = GhostModule(filter[1]*3, filter[1])
        # self.conv8 = Conv11 (filter[1]*3, filter[1])
        self.up9 = nn.ConvTranspose2d(filter[1], filter[0], 2, stride=2,groups=8)
        self.conv9 = GhostModule(filter[0]*3, filter[0])
        # self.conv9 = Conv11 (filter[0]*3, filter[0])
        self.conv10 = nn.Conv2d(filter[0], output_ch, 1)

        self.gau_1 = GAU(filter[4],filter[3])
        self.gau_2 = GAU(filter[3],filter[2])
        self.gau_3 = GAU(filter[2],filter[1])
        self.gau_4 = GAU(filter[1],filter[0])


    def forward(self, x):

        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # c5 = self.local(c5)
        # c5 = self.res2(c5)
        # c5 = self.rfb(c5)
        c5 = self.dropout_layer(c5)

        #print(c5.shape)
        up_6 = self.up6(c5)

        gau1 = self.gau_1(c5,c4)

        merge6 = torch.cat([c4,up_6, gau1], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        gau2 = self.gau_2(gau1,c3)
        merge7 = torch.cat([c3,up_7, gau2], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        gau3 = self.gau_3(gau2,c2)
        merge8 = torch.cat([c2,up_8, gau3], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        gau4 = self.gau_4(gau3,c1)
        merge9 = torch.cat([c1,up_9, gau4], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10
