# Inductor FX Passes

This directory contains the FX passes of Inductor. FX passes are transformations applied to the FX graph to optimize and modify it for better performance and functionality.

In TorchAO, you can replace the following customized graph passes of Inductor:
- `pre_grad_custom_pass`
- `joint_custom_pre_pass`
- `joint_custom_post_pass`
- `post_grad_custom_post_pass`
- `post_grad_custom_pre_pass`

## Directory Structure

- `int8_sdpa_fusion`: Pattern match for int8 sdpa fusion.

## Getting Started

To get started with using the FX passes in TorchAO, you can register and apply them to your FX graph as follows:

```python
from torch._inductor import config
from torch._inductor.pattern_matcher import PatternMatcherPass

# Example usage
patterns = PatternMatcherPass() # create a pattern matcher pass
_register_patterns(...) # register your own patterns
config.custom_pass = patterns.apply # define the custom pass with the patterns

```

## Limitations

For now, we can only register one pass as the custom pass.
In the future, it is better to extend it to a list.
