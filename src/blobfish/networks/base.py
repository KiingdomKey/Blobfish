"""
Base network for blobfish

Every architecture we try shall be built on top of this generic.
This is the contract which says what goes in and what comes out so mcts,
self play and training never have to deal directly with the network they're using

in:  board planes determined by an encoding file, shape (batch, 18, size, size)
out: (policy_logits, value)
     policy_logits is just (batch, number of move slots), one raw score per move
     value is (batch, 1), how good the position is from -1 for losing to 1 for winning

training batches are dicts with:
     state   (batch, 18, size, size)  the encoded boards
     policy  (batch, move slots)      what the policy should have said, mcts visit counts or a one-hot of the move played
     value   (batch,) or (batch, 1)   what the value should have said, game result or mcts value from -1 to 1
     mask    (batch, move slots)      optional, 1 for legal moves 0 for illegal

usage:
     net = make_network("resnet", input_shape=(18, 5, 5), output_shape=(policy_size(5),))
     lightning.Trainer(max_epochs=1).fit(net, dataloader_of_dicts)
     probs, values = net.predict(boards, mask=legal_masks)
"""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F


# There are a butt-ton of comments below for Rachel in case you do not know the deal
# with any neural nets jargon

# SOME SERIOUS VOCABS:
#
#   tensor - similar to a numpy array but it can run on the gpu and track gradients
#
#   batch  - we're gonna feed the network lotsa boards at once, and the first dimension of every
#            tensor will be the batch, so (batch, 18, 5, 5) means "batch of boards,
#            each 18x5x5" (or 6x6 or 8x8 depending)
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
                 lr = 1e-3,
                 weight_decay = 1e-4,
                 value_weight = 1.0,
                 policy_loss = "cross_entropy",
                 **kwargs):
        # This stuff is all setup nonsense
        super().__init__(**kwargs)

        # remember every init argument (including the subclass ones like channels and blocks)
        # so a saved checkpoint can rebuild the exact same network with load_from_checkpoint
        # a checkpoint is just the trained weights saved to a file so we dont lose hours of training
        self.save_hyperparameters()

        # training knobs
        # lr is the learning rate aka how big of a step gradient descent takes each time
        # weight_decay is l2 regularization, same thing as the penalty in ridge regression
        # value_weight says how much the value loss counts next to the policy loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.value_weight = value_weight

        # which policy loss to use
        #   cross_entropy - alphazero style, match the mcts move probabilities (log loss with soft labels)
        #   advantage     - actor critic style, same log loss but each board is weighted by how much better
        #                   it went than the critic expected so surprising wins get pushed up harder
        if policy_loss not in ("cross_entropy", "advantage"):
            raise ValueError(f"policy_loss must be 'cross_entropy' or 'advantage', got '{policy_loss}'")
        self.policy_loss = policy_loss

        # save the shapes so subclasses can use them to size their layers
        # ex for 5x5 input_shape is (18, 5, 5) and output_shape is (1225,)
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

    @staticmethod
    def mask_logits(p, mask):
        # most of the move slots are illegal in any given position (a pawn cant move like a knight duh)
        # so we give illegal moves a score of -infinity and softmax turns that into exactly 0 probability
        # no mask means we dont know whats legal so we leave the scores alone
        if mask is None:
            return p

        # if a board has no legal moves at all (game over) we leave it alone too
        # otherwise every score is -infinity and softmax divides by 0 and spits out nans (not a number)
        mask = mask.to(device=p.device, dtype=torch.bool)
        has_legal = mask.any(dim=-1, keepdim=True)
        return p.masked_fill(~mask & has_legal, float("-inf"))

    def predict(self, x, mask=None):
        # self(x) is how you run a pytorch network, it calls forward for us
        p, v = self(x)

        # mcts only cares about legal moves so if we know which ones are legal we zero out the rest
        p = self.mask_logits(p, mask)

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
        # a batch can also be a dict with a mask if we want legal move probabilities
        if isinstance(predict_batch, dict):
            return self.predict(predict_batch["state"], predict_batch.get("mask"))
        return self.predict(predict_batch)

    @torch.inference_mode()
    def evaluate(self, x, mask=None):
        # the one call mcts needs, boards in and (move probabilities, values) out
        # inference_mode up top turns off gradient tracking since we're just predicting not learning
        # which makes it faster and uses less memory

        # eval mode makes batchnorm use its saved averages instead of the current batch so a board
        # gets the same answer no matter what other boards are in the batch, then we put it back how it was
        was_training = self.training
        self.eval()
        p, v = self.predict(x.to(self.device, torch.float32), mask)
        self.train(was_training)

        # value comes back as (batch,) so each mcts thread can grab its own number
        return p, v.squeeze(-1)

    def compute_loss(self, batch):
        # this is the score for how wrong the network is, training is just making this number go down
        # training_step and validation_step both use it so it lives in one spot

        # run the boards through, squeeze turns the value from (batch, 1) into (batch,)
        # and the targets get matched to the same shape and type as the outputs so the math lines up
        p, v = self(batch["state"])
        v = v.squeeze(-1)
        value_target = batch["value"].to(v.dtype).reshape(v.shape)
        policy_target = batch["policy"].to(p.dtype)

        # value loss is plain mean squared error since the value head is a regression
        value_loss = F.mse_loss(v, value_target)

        # policy loss is log loss against the target probabilities, same as logistic regression
        # but the labels can be spread over a few moves instead of being one right answer
        # log_softmax is log(softmax) done in one numerically safe step, and masked moves become -inf
        # so we swap those for 0 since their target is 0 anyway and 0 * -inf would be nan
        log_probs = F.log_softmax(self.mask_logits(p, batch.get("mask")), dim=-1)
        log_probs = torch.where(policy_target > 0, log_probs, torch.zeros_like(log_probs))
        per_board = -(policy_target * log_probs).sum(dim=-1)

        # for actor critic we weight each board by its advantage, how much better the result was than
        # the critic guessed, detach means dont let this weight train the critic (that's the value loss job)
        if self.policy_loss == "advantage":
            advantage = (value_target - v).detach()
            per_board = per_board * advantage

        # average over the batch so the loss doesnt get bigger just because the batch did
        policy_loss = per_board.mean()

        # one number for lightning to minimize, both heads get trained at the same time
        loss = policy_loss + self.value_weight * value_loss
        return loss, policy_loss, value_loss

    def training_step(self, batch, batch_idx):
        # lightning calls this for every batch during trainer.fit, we hand back the loss and it does
        # the backward pass (figuring out the gradients) and the optimizer step (nudging the weights) for us
        loss, policy_loss, value_loss = self.compute_loss(batch)

        # log the losses so we can watch them go down, the policy and value ones separately
        # so we can tell which head is struggling
        n = batch["state"].shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=n)
        self.log("train/policy_loss", policy_loss, batch_size=n)
        self.log("train/value_loss", value_loss, batch_size=n)
        return loss

    def validation_step(self, batch, batch_idx):
        # same loss on boards we dont train on so we can tell if its actually learning or just memorizing
        # basically a test set score, no weights get changed here
        loss, policy_loss, value_loss = self.compute_loss(batch)
        n = batch["state"].shape[0]
        self.log("val/loss", loss, prog_bar=True, batch_size=n)
        self.log("val/policy_loss", policy_loss, batch_size=n)
        self.log("val/value_loss", value_loss, batch_size=n)

    def configure_optimizers(self):
        # the optimizer is what actually changes the weights using the gradients
        # adamw is gradient descent that picks its own step size for each weight, plus the weight decay from above
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
