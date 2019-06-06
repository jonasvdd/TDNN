import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

__author__ = 'Jonas Van Der Donckt'


class TDNN_ASYM(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, context: list):
        """
        Implementation of a TDNN layer which uses weight masking to create non symmetric convolutions

        :param input_channels: The number of input channels
        :param output_channels: The number of channels produced by the temporal convolution
        :param context: The temporal context
        """
        super(TDNN_ASYM, self).__init__()

        # create the convolution mask
        self.conv_mask = self._create_conv_mask(context)

        # TDNN convolution
        self.temporal_conv = weight_norm(nn.Conv1d(input_channels, output_channels,
                                                   kernel_size=self.conv_mask.size()[0]))

        # expand the mask and register a hook to zero gradients during backprop
        self.conv_mask = self.conv_mask.expand_as(self.temporal_conv.weight)
        self.temporal_conv.weight.register_backward_hook(lambda grad: grad * self.conv_mask)

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_channels, sequence_length]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, output_dim, sequence_length - kernel_size + 1]
        """
        return self.temporal_conv(x)

    @staticmethod
    def _create_conv_mask(context: list) -> torch.Tensor:
        """
        :param context: The temporal context
            TODO some more exlanation about the convolution
        :return: The convolutional mask
        """
        context = sorted(context)
        min_pos, max_pos = context[0], context[-1]
        mask = torch.zeros(size=(max_pos - min_pos + 1,), dtype=torch.int8)
        context = list(map(lambda x:  x-min_pos, context))
        mask[context] = 1
        return mask
