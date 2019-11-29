""" sdss_unet
=====

Implements the :class:`~picasso.ML.SDSSUNet` class which is a
ResBlockUnet with quantile regression. 
This network is optimized for padded SDSS images of size 256x256 pixels.

"""

from inferno.extensions.models.res_unet import _ResBlock, ResBlockUNet

import numpy as np
import torch
import torch.nn as nn
from . import UNet2dGN


class SDSSUNet(UNet2dGN):
    """
        A ResBlockUnet with quantile regression
        This ds is optimized for padded SDSS images (256, 256)

        Parameters:

            quantiles:  list of length out_channels that specifies the quantiles
                        for every channel. One element of quantiles is a ordered
                        list of real values that give the cornerstones of
                        the quantization.
        """

    def __init__(self, in_channels, out_channels, dim, quantiles, first_kernel=9, **kwargs):

        self.quantile_steps = [[p - q for q, p in zip(cq[:-1], cq[1:])] for cq in quantiles]
        self.quantile_steps = torch.from_numpy(np.array(self.quantile_steps)).reshape(1, out_channels, -1, 1, 1)
        self.quantile_steps.requires_grad = False
        # todo make cuda conditional
        self.quantile_steps = self.quantile_steps.float().cuda()

        self.quantile_start = [cq[:-1] for cq in quantiles]
        self.quantile_start = torch.from_numpy(np.array(self.quantile_start)).reshape(1, out_channels, -1, 1, 1)
        self.quantile_start.requires_grad = False
        self.quantile_start = self.quantile_start.float().cuda()

        self.num_quantiles = self.quantile_steps.shape[2]
        self.all_quantiles = self.num_quantiles * out_channels
        self.final_out_channels = out_channels

        self.first_kernel = first_kernel

        print(kwargs)
        super(SDSSUNet, self).__init__(in_channels, 2 * out_channels * self.num_quantiles,
                                       final_activation=None, **kwargs["unet_kwargs"])

    def forward(self, input):
        out = super(SDSSUNet, self).forward(input)
        outshape = list(out[:, None, None].shape)
        outshape[1] = 2
        outshape[2] = self.final_out_channels
        outshape[3] = self.num_quantiles
        out = out.view(tuple(outshape))
        # calculate the softmax weighting function for the quantile regressors
        quantiles = nn.functional.softmax(out[:, 0], dim=2)

        # add quantile base to the residual regressors
        residuals = nn.functional.sigmoid(out[:, 1])
        residuals = residuals * self.quantile_steps
        residuals = residuals + self.quantile_start

        return (residuals * quantiles).sum(dim=2)

    def _upsampler(self, in_channels, out_channels, level):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             nn.Conv2d(in_channels, out_channels, kernel_size=1))

    # def _forward_sanity_check(self, input):
    #     pass

    # def conv_op_factory(self, in_channels, out_channels, part, index):

    #     # is this the very last convolutional block?
    #     very_last = (part == 'up' and index == 0)
    #     very_first = (part == 'down' and index == 0)

    #     # should the residual block be activated?
    #     activated = not very_last or self.activated

    #     # should the output be part of the overall
    #     # return-list in the forward pass of the UNet
    #     use_as_output = part in self.side_out_parts

    #     conv_kwargs = conv_kwargs = dict(
    #              kernel_size=self.first_kernel, dim=self.dim, activation=None,
    #              stride=1, dilation=1, groups=None, depthwise=False, bias=True,
    #              deconv=False, initialization=None
    #         )

    #     # residual block used within the UNet
    #     if very_first:
    #         return _ResBlock(in_channels=in_channels, out_channels=out_channels,
    #                      dim=self.dim, activated=activated, conv_kwargs=conv_kwargs,
    #                      **self.res_block_kwargs), use_as_output
    #     else:
    #         return _ResBlock(in_channels=in_channels, out_channels=out_channels,
    #                      dim=self.dim, activated=activated,
    #                      **self.res_block_kwargs), use_as_output



class SDSSUNet_legacy(SDSSUNet):
    """
        A vanilla unet with 3, 3, 2, 2, ... down sampling
        This ds is optimized for padded SDSS images (432, 432)
    """

    def downsample_op_factory(self, index):
        kernel_size=3 if index < 2 else 2
        C=nn.MaxPool2d if self.dim == 2 else nn.MaxPool3d
        return C(kernel_size=kernel_size, stride=kernel_size)

    def upsample_op_factory(self, index):
        args=dict(self._upsample_kwargs)
        args["scale_factor"]=3 if index < 2 else 2
        return nn.Upsample(**args)
