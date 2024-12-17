from inspect import isfunction
from torch import nn
from torch.nn import init


# Utility: Check if a value exists (not None)
def exists(x):
    return x is not None


# Utility: Return a default value if the primary value does not exist - No use
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# Utility: Create an infinite cycle of a DataLoader - No use
def cycle(dl):
    while True:
        for data in dl:
            yield data


# Utility: Divide a number into groups of a given size
def num_to_groups(num, divisor):
    """
    Divide `num` into groups of size `divisor` and return the group sizes.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# Initialize weights for neural network layers
def initialize_weights(net_l, scale=0.1):
    #compatible with layers handling 502-channel spatial data.
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            # Convolutional layers
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')  # He initialization
                m.weight.data *= scale  # Adjust scale for residual blocks
                if m.bias is not None:
                    m.bias.data.zero_()  # Zero initialize biases

            # Fully connected layers
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()

            # Batch normalization layers
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # Scale weights to 1
                init.constant_(m.bias.data, 0.0)  # Zero initialize biases


# Create a sequence of layers
def make_layer(block, n_layers, seq=False):
# updated to ensurss compatibility with spatial transcriptomics by allowing flexible layer creation
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    # Return as Sequential if seq=True; else as ModuleList
    if seq:
        return nn.Sequential(*layers)
    else:
        return nn.ModuleList(layers)
