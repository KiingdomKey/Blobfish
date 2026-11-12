"""
ResNet for blobfish

A small version of the network alphazero used. One shared trunk of residual blocks reads the board,
then it splits into two heads, policy (which move) and value (who's winning).
Built on top of PolicyValueNet so it gets all the training stuff from base.py for free

build: input_shape  is ONE board (planes, size, size), ex (18, 5, 5), no batch dimension
       so input_shape[0] is the 18 planes and input_shape[1:] is the board size
       output_shape is (number of move slots,), ex (1225,) for 5x5

in:  board planes from encoding.py, shape (batch, 18, size, size)
     this is what goes into forward, the batch only shows up here
out: (policy_logits, value)
     policy_logits is (batch, number of move slots)
     value is (batch, 1), squashed into -1 to 1

usage:
     net = make_network("resnet", input_shape=(18, 5, 5), output_shape=(policy_size(5),), channels=64, blocks=6)
"""

import numpy as np
import torch

from .base import PolicyValueNet


# MORE VOCABS FOR RACHEL:
#
#   conv layer - slides a little 3x3 window over the board and at every square mixes together
#                the square and its neighbors, kinda like a feature extractor that looks at local
#                patterns (a knight next to a pawn, a rook on an open file) and the same window
#                is reused on every square so it learns patterns not positions
#
#   channels   - how many different patterns each conv layer looks for, our board comes in
#                with 18 (one per plane) and the trunk turns that into 64 learned ones
#
#   batchnorm  - rescales each channel to mean 0 and std 1 across the batch, same idea as
#                StandardScaler in sklearn but inside the network, it makes training way less finicky
#
#   relu       - max(0, x), the nonlinearity, without it stacking layers would just be one
#                big linear regression
#
#   residual   - the block adds its input back onto its output (x + y), so each block only has to
#                learn a small correction instead of redoing everything, that's what lets us stack
#                lots of them without training falling apart


# one residual block, the trunk is just a stack of these
class ResidualConv2dLayer(torch.nn.Module):
    def __init__(self,
                 size,
                 **kwargs):
        super().__init__(**kwargs)

        # two conv layers each followed by a batchnorm
        # padding of 1 keeps the board the same size so a 5x5 board stays 5x5 the whole way
        # bias is off because batchnorm right after it would cancel it out anyway
        self.conv_layer1 = torch.nn.Conv2d(size,
                                           size,
                                           kernel_size=(3,3),
                                           stride=(1,1),
                                           padding=(1,1),
                                           bias=False)
        self.norm1 = torch.nn.BatchNorm2d(size)
        self.conv_layer2 = torch.nn.Conv2d(size,
                                           size,
                                           kernel_size=(3,3),
                                           stride=(1,1),
                                           padding=(1,1),
                                           bias=False)
        self.norm2 = torch.nn.BatchNorm2d(size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # conv then batchnorm then relu
        y = x
        y = self.conv_layer1(y)
        y = self.norm1(y)
        y = self.activation(y)

        # conv then batchnorm again but hold off on the relu
        y = self.conv_layer2(y)
        y = self.norm2(y)

        # the residual part, add the input back on then relu
        y = x + y
        y = self.activation(y)
        return y


class ResNet(PolicyValueNet):
    def __init__(self,
                 input_shape,
                 output_shape,
                 channels = 64,
                 blocks = 6,
                 **kwargs):
        # base.py saves the shapes and sets up the training knobs
        super().__init__(input_shape=input_shape, output_shape=output_shape, **kwargs)

        # number of squares on the board, 25 for 5x5
        # the heads need it to know how big their flattened input is
        board_cells = int(np.prod(input_shape[1:]))

        # trunk
        # the projection turns the 18 input planes into however many channels we want
        # then a stack of residual blocks does the actual thinking about the position
        self.projection = torch.nn.Conv2d(input_shape[0],
                                          channels,
                                          kernel_size=(3,3),
                                          stride=(1,1),
                                          padding=(1,1),
                                          bias=False)
        self.projection_norm = torch.nn.BatchNorm2d(channels)
        self.conv_activation = torch.nn.ReLU()
        self.hidden = torch.nn.Sequential(*[ResidualConv2dLayer(channels) for _ in range(blocks)])

        # flatten turns (batch, channels, size, size) into (batch, channels * size * size)
        # so a normal linear layer (aka logistic or linear regression) can read it
        self.flatten = torch.nn.Flatten()

        # policy head
        # a 1x1 conv squishes the channels down to 2 per square then a linear layer gives one score per move slot
        self.policy_conv = torch.nn.Conv2d(channels, 2, kernel_size=(1,1))
        self.policy_output = torch.nn.Linear(2 * board_cells, output_shape[-1])

        # value head
        # squish down to 1 channel per square, a small hidden layer, then one number
        # tanh squashes that number into -1 to 1 like a sigmoid but centered on 0
        self.value_conv = torch.nn.Conv2d(channels, 1, kernel_size=(1,1))
        self.value_hidden = torch.nn.Linear(board_cells, 64)
        self.value_output = torch.nn.Linear(64, 1)
        self.value_activation = torch.nn.Tanh()

    def forward(self, x):
        # trunk, shared by both heads so they learn from the same understanding of the board
        y = x
        y = self.projection(y)
        y = self.projection_norm(y)
        y = self.conv_activation(y)
        y = self.hidden(y)

        # policy head
        # no softmax at the end on purpose, the loss wants raw logits (see predict in base.py)
        p = self.policy_conv(y)
        p = self.conv_activation(p)
        p = self.flatten(p)
        p = self.policy_output(p)

        # value head
        v = self.value_conv(y)
        v = self.conv_activation(v)
        v = self.flatten(v)
        v = self.value_hidden(v)
        v = self.conv_activation(v)
        v = self.value_output(v)
        v = self.value_activation(v)
        return p, v
