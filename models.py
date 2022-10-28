from doctest import OutputChecker
from turtle import forward
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class DepthwiseSeperableConv3D(nn.Module):
    '''
    Depthwise Seperable 3D Convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=out_channels)
        self.pointwise_conv = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

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
        self.module.append(nn.AdaptiveAvgPool3d((1, 1, 1))) #channelwise average pooling
        self.module.append(nn.Linear(n_channels, n_channels // 8))
        self.module.append(Act())
        self.module.append(nn.Linear(n_channels // 8, n_channels))
        self.module.append(nn.Sigmoid())

    def forward(self, x):
        channel_weights = x
        for layer in self.module():
            channel_weights = layer(channel_weights)
        return x * channel_weights


class SqueezeAndExcitationBlock(nn.Module):
    """
    Squeeze and Excitation Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, layers=5):
        super().__init__()
        assert layers >= 2 # input / output layer + n intermediate layers
        self.input_layer = DepthwiseSeperableConv3D(in_channels, out_channels, kernel_size, 1, padding, dilation)
        self.output_layer = DepthwiseSeperableConv3D(out_channels, out_channels, kernel_size, stride, padding, dilation)
        self.input_bn = nn.BatchNorm3d(out_channels)
        self.output_bn = nn.BatchNorm3d(out_channels)

        self.intermediate_layers = nn.ModuleList()
        for _ in range(layers - 2):
            self.intermediate_layers.append(DepthwiseSeperableConv3D(out_channels, out_channels, kernel_size, 1, padding, dilation))
            self.intermediate_layers.append(nn.BatchNorm3d(out_channels))
            self.intermediate_layers.append(Act())
        # SE itself still not implemented
        self.se = SqueezeAndExcitation(out_channels)

        if stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(in_channels, out_channels)

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
    def __init__(self, initial_channels, kernel_size, stride=1, padding=0, dilation=1, blocks=4, block_layers=5) -> None:
        super().__init__()

        # Feel free to also add arguments to __init__ if you want.
        # ----------------------- ADD YOUR CODE HERE --------------------------
        self.module = nn.ModuleList()
        self.module.append(nn.Conv3d(1, initial_channels, kernel_size=1)) # initial Conv

        for i in range(blocks):
            self.module.append(SqueezeAndExcitationBlock(in_channels=initial_channels//2**i, out_channels=initial_channels//2**(i+1), kernel_size=kernel_size,\
                 stride=stride, padding=padding, dilation=dilation, layers=block_layers))

        self.module.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.module.append(nn.Linear(initial_channels // 2**blocks, 1))
        # ------------------------------- END ---------------------------------

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Forward pass of your model.

        :param imgs: Batch of input images. Shape (N, 1, H, W, D)
        :return pred: Batch of predicted ages. Shape (N)
        """
        # ----------------------- ADD YOUR CODE HERE --------------------------
        pred = imgs
        for layer in self.module:
            pred = layer(pred)
        # ------------------------------- END ---------------------------------
        return pred.squeeze()

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
        loss = nn.SmoothL1Loss()(pred, labels)
        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss
