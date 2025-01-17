import abc


# directory location for this might need more polish
class AOBaseWorkflowConfig(abc.ABC):
    """
    If a workflow config inherits from this then `quantize_` knows
    what to do with it.

    TODO write a better docblock.
    """

    pass
