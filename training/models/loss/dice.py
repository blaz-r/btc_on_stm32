"""Dice loss implementation."""


def dice_loss_smooth(inputs, targets):
    """
    Computes the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    smooth = 1
    inputs = inputs.sigmoid()
    inputs = inputs.flatten()
    targets = targets.flatten()
    numerator = 2 * (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()
    loss = 1 - (numerator + smooth) / (denominator + smooth)
    return loss


def dice_loss_eps(inputs, targets, naive=True, eps=1e-3):
    """
    Computes the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten()
    targets = targets.flatten()
    numerator = 2 * (inputs * targets).sum()
    if naive:
        denominator = inputs.sum() + targets.sum()
    else:
        denominator = (inputs * inputs).sum() + (targets * targets).sum()
    d = numerator / (denominator + eps)
    loss = 1 - d
    return loss
