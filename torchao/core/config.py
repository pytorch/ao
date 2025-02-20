import abc


class AOBaseConfig(abc.ABC):
    """
    If a workflow config inherits from this then `quantize_` knows
    how to a apply it to a model. For example::

        # user facing code
        class WorkflowFooConfig(AOBaseConfig): ...
            # configuration for workflow `Foo` is defined here
            bar = 'baz'

        # non user facing code
        @register_quantize_module_handler(WorkflowFooConfig)
        def _transform(
            mod: torch.nn.Module,
            config: WorkflowFooConfig,
        ) -> torch.nn.Module:
            # the transform is implemented here, usually a tensor sublass
            # weight swap or a module swap
            ...

        # then, the user calls `quantize_` with a config, and `_transform` is called
        # under the hood by `quantize_.

    """

    pass
