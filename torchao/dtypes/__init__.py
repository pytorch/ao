# TODO: remove this stub once external repos have migrated off AffineQuantizedTensor
class AffineQuantizedTensor:
    """
    Stub for BC. AQT has been deleted; this ensures isinstance() checks
    return False without errors.

    As of diffusers 0.37.1 and transformers 5.5.0, both libraries have
    code like the following:

        from torchao.dtypes import AffineQuantizedTensor
        if isinstance(weight, AffineQuantizedTensor):
            return f"{weight.__class__.__name__}({weight._quantization_type()})"

    This stub keeps that code working without errors. References to AQT
    are being removed in these libraries:
    - https://github.com/huggingface/diffusers/pull/13405
    - https://github.com/huggingface/transformers/pull/45321
    """

    pass


__all__ = [
    "AffineQuantizedTensor",
]
