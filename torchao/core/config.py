import torch


# directory location for this might need more polish
class ao_base_workflow_config:
    """
    If a workflow config inherits from this then `quantize_` knows
    what to do with it.

    Name is underscore-case to match existing user facing quantize_ workflow API
    names.
    """

    def _transform(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Defines how to apply the configured transformation to the module,
        returns the modified module. This function will be called in the internals of the
        `quantize_` API.

        THIS IS NOT A PUBLIC API - any usage of this outside of torchao
        can break at any time.

        Vasiliy: `_transform` is currently here to keep the underlying code
        colocated with the config, matching existing code structure. However,
        IMO it would be cleaner to move this to a private location, allowing the
        config object to be 100% clean of implementation details, which will make
        it easier to understand for users. Saving this for a future,
        optional conversation.

        TODO(before land): polish this docblock
        """
        raise NotImplementedError()
