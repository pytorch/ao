import abc


class AOBaseConfig(abc.ABC):
    """
    If a workflow config inherits from this then `quantize_` knows
    how to a apply it to a model.
    """

    pass
