# Import the utility modules first
from .utils import (
    Layout,
    PlainLayout,
)

# Import low-level ops that don't depend on tensor subclasses
from . import affine_quantized_tensor_ops

# Import tensor subclasses and conversion functions, but use import statements inside functions 
# to break circular dependencies. These will be lazy-loaded when accessed.
def _import_affine_quantized_tensor():
    from .affine_quantized_tensor import (
        AffineQuantizedTensor,
        to_affine_quantized_floatx,
        to_affine_quantized_floatx_static,
        to_affine_quantized_fpx,
        to_affine_quantized_intx,
        to_affine_quantized_intx_static,
    )
    return (
        AffineQuantizedTensor,
        to_affine_quantized_floatx,
        to_affine_quantized_floatx_static,
        to_affine_quantized_fpx,
        to_affine_quantized_intx,
        to_affine_quantized_intx_static,
    )

def _import_nf4tensor():
    from .nf4tensor import NF4Tensor, to_nf4
    return NF4Tensor, to_nf4

def _import_floatx():
    from .floatx import (
        CutlassSemiSparseLayout,
        Float8Layout,
    )
    return CutlassSemiSparseLayout, Float8Layout

def _import_uintx():
    from .uintx import (
        BlockSparseLayout,
        CutlassInt4PackedLayout,
        Int4CPULayout,
        Int4XPULayout,
        MarlinQQQLayout,
        MarlinQQQTensor,
        MarlinSparseLayout,
        PackedLinearInt8DynamicActivationIntxWeightLayout,
        QDQLayout,
        SemiSparseLayout,
        TensorCoreTiledLayout,
        UintxLayout,
        to_marlinqqq_quantized_intx,
    )
    return (
        BlockSparseLayout,
        CutlassInt4PackedLayout,
        Int4CPULayout,
        Int4XPULayout,
        MarlinQQQLayout,
        MarlinQQQTensor,
        MarlinSparseLayout,
        PackedLinearInt8DynamicActivationIntxWeightLayout,
        QDQLayout,
        SemiSparseLayout,
        TensorCoreTiledLayout,
        UintxLayout,
        to_marlinqqq_quantized_intx,
    )

# Create properties for each of the exports
# When accessed for the first time, they'll trigger the import
# and cache the result
_cache = {}

def __getattr__(name):
    # Define which modules to import for each attribute
    import_map = {
        "AffineQuantizedTensor": _import_affine_quantized_tensor,
        "to_affine_quantized_floatx": _import_affine_quantized_tensor,
        "to_affine_quantized_floatx_static": _import_affine_quantized_tensor,
        "to_affine_quantized_fpx": _import_affine_quantized_tensor,
        "to_affine_quantized_intx": _import_affine_quantized_tensor,
        "to_affine_quantized_intx_static": _import_affine_quantized_tensor,
        "NF4Tensor": _import_nf4tensor,
        "to_nf4": _import_nf4tensor,
        "CutlassSemiSparseLayout": _import_floatx,
        "Float8Layout": _import_floatx,
        "BlockSparseLayout": _import_uintx,
        "CutlassInt4PackedLayout": _import_uintx,
        "Int4CPULayout": _import_uintx,
        "Int4XPULayout": _import_uintx,
        "MarlinQQQLayout": _import_uintx,
        "MarlinQQQTensor": _import_uintx,
        "MarlinSparseLayout": _import_uintx,
        "PackedLinearInt8DynamicActivationIntxWeightLayout": _import_uintx,
        "QDQLayout": _import_uintx,
        "SemiSparseLayout": _import_uintx,
        "TensorCoreTiledLayout": _import_uintx,
        "UintxLayout": _import_uintx,
        "to_marlinqqq_quantized_intx": _import_uintx,
    }
    
    if name in import_map:
        if name not in _cache:
            # Get the import function for this name
            import_func = import_map[name]
            
            # Call the import function and store all results in cache
            values = import_func()
            
            # Map the values to their names based on the import function
            if import_func == _import_affine_quantized_tensor:
                names = ["AffineQuantizedTensor", "to_affine_quantized_floatx", 
                         "to_affine_quantized_floatx_static", "to_affine_quantized_fpx", 
                         "to_affine_quantized_intx", "to_affine_quantized_intx_static"]
            elif import_func == _import_nf4tensor:
                names = ["NF4Tensor", "to_nf4"]
            elif import_func == _import_floatx:
                names = ["CutlassSemiSparseLayout", "Float8Layout"]
            elif import_func == _import_uintx:
                names = ["BlockSparseLayout", "CutlassInt4PackedLayout", "Int4CPULayout", 
                         "Int4XPULayout", "MarlinQQQLayout", "MarlinQQQTensor", 
                         "MarlinSparseLayout", "PackedLinearInt8DynamicActivationIntxWeightLayout", 
                         "QDQLayout", "SemiSparseLayout", "TensorCoreTiledLayout", 
                         "UintxLayout", "to_marlinqqq_quantized_intx"]
            
            # Cache all the imported values
            for n, v in zip(names, values):
                _cache[n] = v
        
        return _cache[name]
    
    raise AttributeError(f"module 'torchao.dtypes' has no attribute '{name}'")

__all__ = [
    # These will be loaded lazily via __getattr__
    "NF4Tensor",
    "to_nf4",
    "AffineQuantizedTensor",
    "to_affine_quantized_intx",
    "to_affine_quantized_intx_static",
    "to_affine_quantized_fpx",
    "to_affine_quantized_floatx",
    "to_affine_quantized_floatx_static",
    "to_marlinqqq_quantized_intx",
    # These are directly imported
    "Layout",
    "PlainLayout",
    # These will be loaded lazily via __getattr__
    "SemiSparseLayout",
    "TensorCoreTiledLayout",
    "Float8Layout",
    "MarlinSparseLayout",
    "affine_quantized_tensor_ops",
    "BlockSparseLayout",
    "UintxLayout",
    "MarlinQQQTensor",
    "MarlinQQQLayout",
    "Int4CPULayout",
    "CutlassInt4PackedLayout",
    "CutlassSemiSparseLayout",
    "QDQLayout",
    "PackedLinearInt8DynamicActivationIntxWeightLayout",
    "to_affine_quantized_packed_linear_int8_dynamic_activation_intx_weight",
    "Int4XPULayout",
]
