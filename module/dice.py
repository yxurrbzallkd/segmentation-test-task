import tensorflow as tf
import numpy as np

def dice_coeff(inp: np.array, target: np.array, epsilon: float = 1e-6) -> tf.Tensor:
    """
    Args:
        - inp: predicted mask
        - target: gt mask
        - epsilon: zero division safeguard (0 / (0 + 0 + epsilon))
        
    Returns:
        dice cefficient
    """
    # Average of Dice coefficient for a single mask
    return 2*tf.math.reduce_sum(tf.cast(inp > 0.5,  tf.float32) * target) / (tf.math.reduce_sum(inp) + tf.math.reduce_sum(target) + epsilon)

def dice_loss(inp: np.array, target: np.array, epsilon: float = 1e-6) -> tf.Tensor:
    """ Calculate dice loss
    
    Args:
        - inp:     predicted mask
        - target:  gt mask
        - epsilon: zero division safeguard (0 / (0 + 0 + epsilon))
        
    Returns:
        dice loss
    
    loss = 1 - coefficient
    because
    dice coefficient tells how well the prediction matches the target
    1 - dice_coeff - tells how much off the prediction is
    """
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(inp, target)
