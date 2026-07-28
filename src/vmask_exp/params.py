from __future__ import annotations

import copy


def clone_model(model):
    return copy.deepcopy(model)


def parameter_vector(model):
    import torch
    from torch.nn.utils import parameters_to_vector

    with torch.no_grad():
        return parameters_to_vector(model.parameters()).detach().clone()


def add_update_(model, update, scale: float) -> None:
    import torch
    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    with torch.no_grad():
        current = parameters_to_vector(model.parameters()).detach()
        next_vector = current + scale * update.to(current.device)
        vector_to_parameters(next_vector, model.parameters())
