"""
Base network for blobfish

Every architecture we try shall be built on top of this generic.
This is the contract which says what goes in and what comes out so mcts,
self play and training never have to deal directly with the network they're using

in:  board planes determined by an encoding file, shape (batch, 12, size, size)
out: (policy_logits, value)
     policy_logits is just (batch, number of move slots), one raw score per move
     value is (batch, 1), how good the position is from -1 for losing to 1 for winning
     
"""

import lightning.pytorch as pl
import torch


# There are a butt-ton of comments below for Rachel in case you do not know the deal
# with any neural nets jargon

# SOME SERIOUS VOCABS:
#
#   tensor - similar to a numpy array but it can run on the gpu and track gradients
#
#   batch  - we're gonna feed the network lotsa boards at once, and the first dimension of every
#            tensor will be the batch, so (batch, 12, 5, 5) means "batch of boards,
#            each 12x5x5" (or 6x6 or 8x8 depending)
#
#   logits - raw scores, can be any number, bigger means the model likes it more
#            same thing as the output of logistic regression. we will do an activation to it


# THE LONG AND SKINNY:
# pytorch networks are just classes that inherit from torch.nn.Module and define forward(x),
# which is like a prediction function, similar to a predict() on an sklearn model
# LightningModule is a library on top of pytorch that does the training loop so we don't gotta
# (looping over batches, moving to the gpu, logging) so we only write juicy stuff we care about
# (training_step, configure_optimizers, predict_step)
# When we write the training code it goes here and every architecture gets it for freee!
class PolicyValueNet(pl.LightningModule):
    def __init__(self,
                 input_shape,
                 output_shape,
                 **kwargs):
        # This stuff is all setup nonsense
        super().__init__(**kwargs)

        # save the shapes so subclasses can use them to size their layers
        # ex for 5x5 input_shape is (12, 5, 5) and output_shape is (1225,)
        # (1225 is every possible move slot on a 5x5 board)
        self.input_shape = input_shape
        self.output_shape = output_shape

        # softmax is a fun thing to say and the activation that works best for multi-classification stuff
        # it turns raw policy scores into probabilities that are all positive and add up to 1
        # dim=-1 means do it across the last dimension (the moves), separately for each board in the batch
        # its only used in predict not in forward
        self.policy_activation = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # this is a placeholder and must be defined in specific model files
        # if someone forgets WEEWOOWEEWOO there's an error
        #
        # whatever we write has to return (policy_logits, value) in the shapes at the top of this file
        #
        # the network has two outputs that share everything before them, which are:
        #
        #   policy - which move to play, basically a classifier where every possible move is a class
        #
        #   value  - who is winning, basically a regression with one number squashed into -1 to 1
        raise NotImplementedError

    def predict(self, x):
        # self(x) is how you run a pytorch network, it calls forward for us
        p, v = self(x)

        # forward gives raw scores (logits) on purpose because the training loss wants raw junkz

        # the loss function for the policy is cross entropy and pytorch's version applies softmax
        # inside itself in a very super numerically stable way BUT mcts needs actual probabilities
        # to decide which moves to search, so predict applies softmax

        # the value already got squashed to -1 to 1 inside forward so it dooooooo not change
        p = self.policy_activation(p)
        return p, v

    def predict_step(self, predict_batch, batch_idx):
        # lightning calls this when we do trainer.predict(blahblahblah) on a whole dataset
        # our batches are just board tensors (no labels or nothin) so we pass them straight through to predict
        # batch_idx is which batch number this is, lightning needs it in the signature even though we dont use it
        return self.predict(predict_batch)
