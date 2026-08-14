"""TorchTitan DeepSeek-V3 expert-weight shapes shared by grouped benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV3ModelConfig:
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


DEEPSEEK_V3_MODEL_CONFIGS = (
    DeepSeekV3ModelConfig("debugmodel", 8, 1, 256, 256),
    DeepSeekV3ModelConfig("16B", 64, 8, 2048, 1408),
    DeepSeekV3ModelConfig("671B", 256, 2, 7168, 2048),
)


def get_deepseek_v3_weight_shapes(
    *, factorized_experts: int | None = None
) -> list[DeepSeekV3WeightShape]:
    """Return TorchTitan w1/w3 and w2 shapes, optionally with a smaller E."""
    shapes = []
    for config in DEEPSEEK_V3_MODEL_CONFIGS:
        experts = factorized_experts or config.local_experts
        shapes.extend(
            (
                DeepSeekV3WeightShape(
                    config.model,
                    "gate/up (w1/w3)",
                    experts,
                    config.moe_hidden_dim,
                    config.dim,
                ),
                DeepSeekV3WeightShape(
                    config.model,
                    "down (w2)",
                    experts,
                    config.dim,
                    config.moe_hidden_dim,
                ),
            )
        )
    return shapes
