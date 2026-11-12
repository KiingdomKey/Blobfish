"""
Network picker for blobfish

Lets mcts, self play, training and benchmarks grab a network by name so they never have to
import a specific architecture file. Swapping resnet for something else is just changing a string

in:  a network name plus whatever that network's __init__ wants
out: a PolicyValueNet ready to go

usage:
     net = make_network("resnet", input_shape=(18, 5, 5), output_shape=(policy_size(5),))

to add a new one write it as a PolicyValueNet subclass in its own file and add it to NETWORKS
"""

from .base import PolicyValueNet
from .resnet import ResNet


# every architecture we have, name -> class
NETWORKS = {
    "resnet": ResNet,
}


def make_network(name, **kwargs):
    # look the name up and build it, yell if its a typo so nobody trains the wrong thing by accident
    if name not in NETWORKS:
        raise ValueError(f"Unknown network '{name}'. Choose from: {', '.join(NETWORKS)}")
    return NETWORKS[name](**kwargs)
