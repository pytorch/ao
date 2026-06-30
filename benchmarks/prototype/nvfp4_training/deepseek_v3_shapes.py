"""TorchTitan DeepSeek-V3 expert-weight shapes shared by grouped benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV3ModelShape:
    model: str
    experts: int
    expert_parallel_degree: int
    dim: int
    moe_hidden_dim: int

    @property
    def local_experts(self) -> int:
        return self.experts // self.expert_parallel_degree


@dataclass(frozen=True)
class DeepSeekV3WeightShape:
    model: str
    projection: str
    experts: int
    m: int
    n: int


DEEPSEEK_V3_MODEL_SHAPES = (
    DeepSeekV3ModelShape("debugmodel", 8, 1, 256, 256),
    DeepSeekV3ModelShape("16B", 64, 8, 2048, 1408),
    # TorchTitan has no registered 236B trainer recipe; assume one-node EP=8.
    DeepSeekV3ModelShape("236B", 160, 8, 5120, 1536),
    DeepSeekV3ModelShape("671B", 256, 2, 7168, 2048),
)


def get_deepseek_v3_weight_shapes(
    *, factorized_experts: int | None = None
) -> list[DeepSeekV3WeightShape]:
    """Return TorchTitan w1/w3 and w2 shapes, optionally with a smaller E."""
    shapes = []
    for model in DEEPSEEK_V3_MODEL_SHAPES:
        experts = factorized_experts or model.local_experts
        shapes.extend(
            (
                DeepSeekV3WeightShape(
                    model.model,
                    "gate/up (w1/w3)",
                    experts,
                    model.moe_hidden_dim,
                    model.dim,
                ),
                DeepSeekV3WeightShape(
                    model.model,
                    "down (w2)",
                    experts,
                    model.dim,
                    model.moe_hidden_dim,
                ),
            )
        )
    return shapes
