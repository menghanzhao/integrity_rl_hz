import numpy as np

def weibull_failure_probability(shape, max_thickness):
    """
    Calculate the Weibull failure probability for a given thickness.
    
    :param thickness: Thickness of the material.
    :param shape: Shape parameter of the Weibull distribution.
    :param scale: Scale parameter of the Weibull distribution.
    :return: Failure probability.
    """
    scale = max_thickness / (-np.log(1e-10)) ** (1 / shape)

    def failure_probability(projected_thickness):
        if projected_thickness <= 1:
            return 1.0
        return np.exp(((20.6-projected_thickness) / scale) ** shape) * (1e-10)
    
    return failure_probability