import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)




class Generator(nn.Module):
    def __init__(self,channels, gf=32):
        super(Generator, self).__init__()

        self.gf = gf
        self.channels = channels

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            return block

        def deconv_block(in_channels, out_channels, kernel_size=2, stride=2, padding=0):
            block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            return block

        # U-Net structure
        self.conv1 = conv_block(self.channels, self.gf)
        self.conv1_2 = conv_block(self.gf, self.gf)

        self.conv2 = conv_block(self.gf, self.gf * 2)
        self.conv2_2 = conv_block(self.gf * 2, self.gf * 2)
        self.conv3 = conv_block(self.gf * 2, self.gf * 4)
        self.conv3_2 = conv_block(self.gf * 4, self.gf * 4)
        self.conv4 = conv_block(self.gf * 4, self.gf * 8)
        self.conv4_2 = conv_block(self.gf * 8, self.gf * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deconv3_1 = deconv_block(self.gf * 8, self.gf * 4)
        self.conv3_3 = conv_block(self.gf * 8, self.gf * 4)
        self.deconv2_1 = deconv_block(self.gf * 4, self.gf * 2)
        self.conv2_3 = conv_block(self.gf * 4, self.gf * 2)
        self.deconv1_1 = deconv_block(self.gf * 2, self.gf)
        self.conv1_3 = conv_block(self.gf * 2, self.gf)

        self.deconv3_2 = deconv_block(self.gf * 8, self.gf * 4)
        self.conv3_4 = conv_block(self.gf * 8, self.gf * 4)
        self.deconv2_2 = deconv_block(self.gf * 4, self.gf * 2)
        self.conv2_4 = conv_block(self.gf * 4, self.gf * 2)
        self.deconv1_2 = deconv_block(self.gf * 2, self.gf)
        self.conv1_4 = conv_block(self.gf * 2, self.gf)

        self.final_layer = nn.Conv2d(self.gf, self.channels, kernel_size=4, padding='same')

    def forward(self, x):
        # Down-sampling
        conv1 = self.conv1(x)
        conv1_2 = self.conv1_2(conv1)
        pool1 = self.pool(conv1_2)

        conv2 = self.conv2(pool1)
        conv2_2 = self.conv2_2(conv2)
        pool2 = self.pool(conv2_2)

        conv3 = self.conv3(pool2)
        conv3_2 = self.conv3_2(conv3)
        pool3 = self.pool(conv3_2)

        conv4 = self.conv4(pool3)
        conv4_2 = self.conv4_2(conv4)

        # Up-sampling
        deconv3_1 = self.deconv3_1(conv4_2)
        concat3_1 = torch.cat([conv3, deconv3_1], dim=1)
        conv3_3 = self.conv3_3(concat3_1)

        deconv2_1 = self.deconv2_1(conv3_3)
        concat2_1 = torch.cat([conv2, deconv2_1], dim=1)
        conv2_3 = self.conv2_3(concat2_1)

        deconv1_1 = self.deconv1_1(conv2_3)
        concat1_1 = torch.cat([conv1, deconv1_1], dim=1)
        conv1_3 = self.conv1_3(concat1_1)

        deconv3_2 = self.deconv3_2(conv4_2)
        concat3_2 = torch.cat([conv3_3, deconv3_2], dim=1)
        conv3_4 = self.conv3_4(concat3_2)

        deconv2_2 = self.deconv2_2(conv3_4)
        concat2_2 = torch.cat([conv2_3, deconv2_2], dim=1)
        conv2_4 = self.conv2_4(concat2_2)

        deconv1_2 = self.deconv1_2(conv2_4)
        concat1_2 = torch.cat([conv1_3, deconv1_2], dim=1)
        conv1_4 = self.conv1_4(concat1_2)
        # Final output
        output = torch.tanh(self.final_layer(conv1_4))
        return output


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, channels,df=64):
        super(Discriminator, self).__init__()

        self.df = df
        self.channels = channels
        height = 256
        width = 256
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, normalization=True):
            """Discriminator layer"""
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding),
                     nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                block.append(nn.InstanceNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.channels, self.df, normalization=False),  # d1: Down-sampling
            *discriminator_block(self.df, self.df * 2),  # d2: Down-sampling
            *discriminator_block(self.df * 2, self.df * 4),  # d3: Down-sampling
            *discriminator_block(self.df * 4, self.df * 8),  # d4: Down-sampling
            nn.Conv2d(self.df * 8, 1, kernel_size=4, stride=1, padding='same')  # validity
        )

    def forward(self, img):
        return self.model(img)