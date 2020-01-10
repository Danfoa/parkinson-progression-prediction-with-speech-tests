import numpy as np


def partial_dMF(x, mf_definition, partial_parameter):
    """Calculates the partial derivative of a membership function at a point x."""
    result = 0.0
    sigma = mf_definition[1]['sigma']
    mean = mf_definition[1]['mean']
    if partial_parameter == 'sigma':
        result = (2./sigma**3) * np.exp(-(((x-mean)**2)/(sigma)**2))*(x-mean)**2
    elif partial_parameter == 'mean':
        result = (2./sigma**2) * np.exp(-(((x-mean)**2)/(sigma)**2))*(x-mean)

    return result