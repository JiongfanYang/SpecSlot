import copy


EVAL_WEIGHT_SOURCES = ("raw", "ema1", "ema2")


def build_eval_state_dict(model, source):
    """Build a complete state dict using raw, EMA1, or EMA2 parameters."""
    if source not in EVAL_WEIGHT_SOURCES:
        raise ValueError(
            f"Unknown evaluation weight source {source!r}; "
            f"expected one of {EVAL_WEIGHT_SOURCES}"
        )

    state_dict = copy.deepcopy(model.state_dict())
    if source == "raw":
        return state_dict

    ema_parameters = getattr(model, f"ema_params{source[-1]}", None)
    named_parameters = list(model.named_parameters())
    if ema_parameters is None:
        raise RuntimeError(f"{source} parameters have not been initialized")
    if len(ema_parameters) != len(named_parameters):
        raise RuntimeError(
            f"{source} contains {len(ema_parameters)} parameters, but the model "
            f"contains {len(named_parameters)}"
        )

    for (name, _parameter), ema_parameter in zip(named_parameters, ema_parameters):
        if name not in state_dict:
            raise KeyError(f"Model parameter {name!r} is missing from state_dict")
        if state_dict[name].shape != ema_parameter.shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: state={tuple(state_dict[name].shape)}, "
                f"{source}={tuple(ema_parameter.shape)}"
            )
        state_dict[name] = ema_parameter.detach().clone()
    return state_dict
