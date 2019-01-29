"""
Utilities to convert model gradients to and from a vector.

Adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/convert_parameters.py
"""

import torch
from torch.nn.utils.convert_parameters import _check_param_device

def gradients_to_vector(parameters):
    r"""Convert gradients to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The gradients of the parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)
        assert hasattr(param, 'grad'), "Param has no grad attribute"

        vec.append(param.grad.view(-1))
    return torch.cat(vec)


def vector_to_gradients(vec, parameters):
    r"""Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()

        if param.grad is None:
            param.grad = torch.zeros_like(param)

        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
