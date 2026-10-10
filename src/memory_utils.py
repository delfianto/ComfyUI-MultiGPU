"""Memory accounting without evaluating computed parametrization properties."""

import torch


def model_parameter_bytes(model: torch.nn.Module) -> int:
    # parameters() deduplicates tied parameters and exposes the original
    # storage of weight_norm/spectral_norm instead of synthesized weights.
    # QuantizedTensor.nbytes reports packed storage; numel()*element_size()
    # reports its logical BF16 shape and can double an FP8 allocation estimate.
    return sum(parameter.nbytes for parameter in model.parameters())
