from doctest import OutputChecker
from turtle import forward
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.convs = nn.Sequential(*[nn.Conv3d(in_channels, out_channels, 1), nn.ReLU(), \
            nn.Conv3d(out_channels, out_channels, 3, stride, padding=1), nn.ReLU(), \
                nn.Conv3d(out_channels, out_channels, 1)])
        if stride == 1:
            self.skip = nn.Identity()
        else: 
            self.skip = nn.Sequential(nn.MaxPool3d(2, 2), nn.Conv3d(in_channels, out_channels, 1))

        
    def forward(self, x):
        pred = self.convs(x)
        return pred + self.skip(x)

class ResNet(nn.Module):
    def __init__(self, channels, blocks_per_pyramide_steps) -> None:
        super().__init__()
        self.input_layer = nn.Conv3d(1, channels, 3, padding='same')
        self.block1 = nn.ModuleList()
        for i in range(blocks_per_pyramide_steps - 1):
            self.block1.append(ResNetBlock(channels, channels))
        self.block1.append(ResNetBlock(channels, channels*2, stride=2))

        self.block2 = nn.ModuleList()
        channels *= 2
        for i in range(blocks_per_pyramide_steps - 1):
            self.block2.append(ResNetBlock(channels, channels))
        self.block2.append(ResNetBlock(channels, channels*2, stride=2))

        channels *= 2
        self.block3 = nn.ModuleList()
        for i in range(blocks_per_pyramide_steps - 1):
            self.block3.append(ResNetBlock(channels, channels))
        self.block3.append(ResNetBlock(channels, channels*2, stride=2))
        
        channels *= 2
        self.block4 = nn.ModuleList()
        for i in range(blocks_per_pyramide_steps - 1):
            self.block4.append(ResNetBlock(channels, channels))
        self.block4.append(ResNetBlock(channels, channels*2, stride=2))

        self.b1 = nn.Sequential(*self.block1)
        self.b2 = nn.Sequential(*self.block2)
        self.b3 = nn.Sequential(*self.block3)
        self.b4 = nn.Sequential(*self.block4)
        self.output = nn.Sequential(*[nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(channels * 2, 1)])

    def forward(self, x):
        x = self.input_layer(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.output(x)

    def train_step(
        self,
        imgs: Tensor,
        labels: Tensor,
        return_prediction: Optional[bool] = False
    ):
        """Perform a training step. Predict the age for a batch of images and
        return the loss.

        :param imgs: Batch of input images (N, 1, H, W, D)
        :param labels: Batch of target labels (N)
        :return loss: The current loss, a single scalar.
        :return pred
        """
        pred = self(imgs)  # (N)

        # ----------------------- ADD YOUR CODE HERE --------------------------
        loss = nn.MSELoss()(pred, labels.to(torch.float32))
        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss



class ToyModel(nn.Module):
    def __init__(self, initial_channels, n_convs=4, n_linear=3) -> None:
        super().__init__()
        conv_layers = nn.ModuleList()
        conv_layers.append(nn.Conv3d(1, initial_channels, 3, bias=False))
        conv_layers.append(nn.ReLU())
        for i in range(n_convs):
            conv_layers.append(nn.Conv3d(initial_channels * 2**i, initial_channels * 2**(i + 1), 3, stride=2, padding=1))
            conv_layers.append(nn.ReLU())
        self.convs = nn.Sequential(*conv_layers)

        self.avg = nn.Sequential(*[nn.AdaptiveAvgPool3d(1), nn.Flatten()])

        mlp_layers = nn.ModuleList()
        for _ in range(n_linear - 1):
            mlp_layers.append(nn.Linear(initial_channels * 2**n_convs, initial_channels * 2**n_convs))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(initial_channels * 2**n_convs, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = self.convs(x)
        x = self.avg(x)
        print(x.shape)
        x = self.mlp(x)
        return x

    def train_step(
        self,
        imgs: Tensor,
        labels: Tensor,
        return_prediction: Optional[bool] = False
    ):
        """Perform a training step. Predict the age for a batch of images and
        return the loss.

        :param imgs: Batch of input images (N, 1, H, W, D)
        :param labels: Batch of target labels (N)
        :return loss: The current loss, a single scalar.
        :return pred
        """
        pred = self(imgs)  # (N)

        # ----------------------- ADD YOUR CODE HERE --------------------------
        loss = nn.SmoothL1Loss()(pred.squeeze(), labels)
        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss


class DepthwiseSeperableConv3D(nn.Module):
    '''
    Depthwise Seperable 3D Convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class Act(nn.Module): 
    """
    Wrapper to experiment with different activation functions like ReLU or Swish
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return nn.ReLU()(x)


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze And Excitation function
    """
    def __init__(self, n_channels) -> None:
        super().__init__()
        self.module = nn.ModuleList()
        self.squeeze = nn.AdaptiveAvgPool3d(1) #channelwise average pooling
        self.excite = nn.Sequential(*[nn.Conv3d(n_channels, n_channels // 8, 1), nn.ReLU(), \
            nn.Conv3d(n_channels // 8, n_channels, 1), nn.Sigmoid()])
    def forward(self, x):
        n, c, h, w, d = x.shape
        channel_weights = self.squeeze(x)
        channel_weights = self.excite(channel_weights).view(n, c, 1, 1, 1)
        return x * channel_weights.expand_as(x)


class SqueezeAndExcitationBlock(nn.Module):
    """
    Squeeze and Excitation Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, layers=5):
        super().__init__()
        assert layers >= 2 # input / output layer + n intermediate layers
        self.input_layer = DepthwiseSeperableConv3D(in_channels, out_channels, kernel_size, 1, padding, dilation)
        self.input_bn = nn.BatchNorm3d(out_channels)

        self.intermediate_layers = nn.ModuleList()
        for _ in range(layers - 2):
            self.intermediate_layers.append(DepthwiseSeperableConv3D(out_channels, out_channels, kernel_size, 1, padding, dilation))
            self.intermediate_layers.append(nn.BatchNorm3d(out_channels))
            self.intermediate_layers.append(Act())

        self.output_layer = DepthwiseSeperableConv3D(out_channels, out_channels, kernel_size, stride, kernel_size//2, dilation)
        self.output_bn = nn.BatchNorm3d(out_channels)

        # SE itself still not implemented
        self.se = SqueezeAndExcitation(out_channels)

        if stride == 1 and out_channels == in_channels:
            self.skip = nn.Identity()
        elif stride == 1:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, padding=0)
        else:
            self.skip = nn.Sequential(nn.MaxPool3d(2,2), nn.Conv3d(in_channels, out_channels, 1, padding=0))

    def forward(self, x):
        skip = self.skip(x)
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = Act()(x)
        for layer in self.intermediate_layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.output_bn(x)
        x = Act()(x)
        x = self.se(x)
        x += skip
        return x

 
class BrainAgeCNN(nn.Module):
    """
    The BrainAgeCNN predicts the age given a brain MR-image.
    """
    def __init__(self, initial_channels, kernel_size, stride=2, padding='same', dilation=1, blocks=4, block_layers=5, input_dim=1) -> None:
        super().__init__()

        # Feel free to also add arguments to __init__ if you want.
        # ----------------------- ADD YOUR CODE HERE --------------------------
        self.se_blocks = nn.ModuleList()
        self.se_blocks.append(nn.Conv3d(input_dim, initial_channels, kernel_size=3, bias=False, padding='same')) # initial Conv

        for i in range(blocks):
            self.se_blocks.append(SqueezeAndExcitationBlock(in_channels=initial_channels * 2**i, out_channels=initial_channels * 2**(i+1), kernel_size=kernel_size,\
                 stride=stride, padding=padding, dilation=dilation, layers=block_layers))
        self.convs = nn.Sequential(*self.se_blocks)
        self.averaging = nn.AdaptiveAvgPool3d(1)
        self.output_mlp = nn.Sequential(nn.Linear(initial_channels * 2**blocks, initial_channels * 2**blocks), nn.ReLU(), nn.Linear(initial_channels * 2**blocks, 1))
        # ------------------------------- END ---------------------------------

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Forward pass of your model.

        :param imgs: Batch of input images. Shape (N, 1, H, W, D)
        :return pred: Batch of predicted ages. Shape (N)
        """
        # ----------------------- ADD YOUR CODE HERE --------------------------
        pred = self.convs(imgs)
        pred = self.averaging(pred)
        pred = nn.Flatten()(pred)
        pred = self.output_mlp(pred)
        # ------------------------------- END ---------------------------------
        return pred

    def train_step(
        self,
        imgs: Tensor,
        labels: Tensor,
        return_prediction: Optional[bool] = False
    ):
        """Perform a training step. Predict the age for a batch of images and
        return the loss.

        :param imgs: Batch of input images (N, 1, H, W, D)
        :param labels: Batch of target labels (N)
        :return loss: The current loss, a single scalar.
        :return pred
        """
        pred = self(imgs)  # (N)

        # ----------------------- ADD YOUR CODE HERE --------------------------
        loss = nn.SmoothL1Loss()(pred.squeeze(), labels)
        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss
