import dataclasses
import re
from typing import Any, Dict, List, Protocol, Tuple, Type

import torch

from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    MappingType,
    # Float8StaticActivationFloat8WeightConfig,
    UIntXWeightOnlyConfig,
)

# Create a type alias for AOBaseConfig classes
ConfigType = Type[AOBaseConfig]


# Define a Protocol for parameter processors
class ParameterProcessor(Protocol):
    """Protocol defining the interface for parameter processors"""

    def __call__(self, match: re.Match, quant_config: ConfigType) -> Tuple[str, Any]:
        """
        Process a regex match into a parameter name and value

        Args:
            match: The regex match object containing captured groups
            quant_config: The quantization config class being instantiated

        Returns:
            Tuple of (parameter_name, parameter_value)

        Note:
            If you need special handling based on the quant_config type,
            be sure to use issubclass instead of isinstance.
        """
        ...


def process_bits(match: re.Match, quant_config: AOBaseConfig) -> Tuple[str, Any]:
    return "bits", int(match.group(1))


def process_group_size(match: re.Match, quant_config: AOBaseConfig) -> Tuple[str, Any]:
    return "group_size", int(match.group(1))


def process_activation_bits(
    match: re.Match, quant_config: AOBaseConfig
) -> Tuple[str, Any]:
    return "activation_bits", int(match.group(1))


def process_weight_bits(match: re.Match, quant_config: AOBaseConfig) -> Tuple[str, Any]:
    return "weight_bits", int(match.group(1))


def process_symmetry(match: re.Match, quant_config: AOBaseConfig) -> Tuple[str, Any]:
    mapping_type = (
        MappingType.SYMMETRIC if match.group(1) == "sym" else MappingType.ASYMMETRIC
    )
    return "mapping_type", mapping_type


def process_dtype(match: re.Match, quant_config: AOBaseConfig) -> Tuple[str, Any]:
    dtype_map = {
        "int4": torch.int4,
        "int8": torch.int8,
        "uint4": torch.uint4,
        "uint8": torch.uint8,
        "e4m3": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2,
    }
    # The float8's have different key names :(
    key = (
        "weight_dtype"
        if issubclass(
            quant_config,
            (
                Float8WeightOnlyConfig,
                Float8DynamicActivationFloat8WeightConfig,
            ),
        )
        else "dtype"
    )
    return key, dtype_map[match.group(1)]


def process_per_row(match: re.Match, quant_config: AOBaseConfig) -> Tuple[str, Any]:
    return "per_row", True


class ConfigParser:
    """Parser for string-based configuration patterns"""

    # Parameter patterns with their processing functions
    param_patterns: Dict[re.Pattern, ParameterProcessor] = {
        re.compile(r"(\d+)bit"): process_bits,
        re.compile(r"g(\d+)"): process_group_size,
        # re.compile(r"act(\d+)"): process_activation_bits,
        # re.compile(r"w(\d+)"): process_weight_bits,
        re.compile(r"(sym|asym)"): process_symmetry,
        re.compile(r"(int4|int8|uint4|uint8|e4m3|e5m2)"): process_dtype,
        re.compile(r"(per_row)"): process_per_row,
    }

    # Map from string prefix to QuantType
    type_mapping = {
        "int4wo": Int4WeightOnlyConfig,
        "int8wo": Int8WeightOnlyConfig,
        "int8dqint4": Int8DynamicActivationInt4WeightConfig,
        "int8dqint8": Int8DynamicActivationInt8WeightConfig,
        "int4dqint4": Int4DynamicActivationInt4WeightConfig,
        "float8wo": Float8WeightOnlyConfig,
        "float8dqfloat8": Float8DynamicActivationFloat8WeightConfig,
        # "float8staticfloat8": Float8StaticActivationFloat8WeightConfig,
        "uintxwo": UIntXWeightOnlyConfig,
        # "fpx": FPXWeightOnlyConfig,
    }

    def parse(self, config_str: str) -> AOBaseConfig:
        """
        Parse a configuration string into an AO quantization configuration object.

        This is the main entrypoint for converting string-based configuration into actual config objects.
        The expected format is "base_param1-value1_param2-value2" where "base" identifies the base
        quantization type and subsequent tokens specify parameter values.

        Examples:
            config_parser.parse("int8dqint8")
            config_parser.parse("int8dqint4_g32")

        Args:
            config_str: String representation of the quantization configuration

        Returns:
            AOBaseConfig: Instantiated quantization configuration object

        Raises:
            ValueError: If the config string is empty or invalid
        """
        tokens = config_str.split("_")

        if not tokens:
            raise ValueError("Empty config string")

        # The first token is the base quantization type
        quant_config = self._get_config(tokens[0])

        # We know the base quant type, now we convert each token to its parameter
        params = self._extract_params(quant_config, tokens[1:])

        return self._instantiate_config(quant_config, params)

    def _get_config(self, first_token: str) -> AOBaseConfig:
        """Get the quantization config from a string"""
        try:
            quant_config = self.type_mapping[first_token]
        except KeyError:
            # Print available base configurations before raising error
            available_configs = list(self.type_mapping.keys())
            raise ValueError(
                f"Unknown quantization type in string: {first_token} \n Available base configurations: {available_configs}"
            )
        return quant_config

    def _instantiate_config(
        self, quant_config: AOBaseConfig, params: Dict[str, Any]
    ) -> AOBaseConfig:
        """Sprinkle some extra logic for helping w/ instantiation failures"""
        try:
            return quant_config(**params)
        except TypeError as e:
            # Get proper field information for error message
            valid_fields = {
                field.name
                for field in dataclasses.fields(quant_config)
                if field.name != "self"
            }
            invalid_params = {k: v for k, v in params.items() if k not in valid_fields}

            field_info = [field.name for field in dataclasses.fields(quant_config)]

            raise ValueError(
                f"Invalid parameters for {quant_config.__name__}: {list(invalid_params.keys())}.\n"
                f"Available parameters for {quant_config.__name__}: {field_info}"
            ) from e

    def _extract_params(
        self, quant_config: AOBaseConfig, param_tokens: List[str]
    ) -> Dict[str, Any]:
        """Extract parameters from tokens"""
        params = {}

        for token in param_tokens:
            if not token:
                continue

            matched = False
            # Try to match against parameter patterns
            # We could specify an ordering but for now we just try all
            for pattern, processor in self.param_patterns.items():
                match = pattern.fullmatch(token)
                if match:
                    param_name, value = processor(match, quant_config)
                    params[param_name] = value
                    matched = True
                    break

            if not matched:
                field_info = [
                    (field.name, field.type)
                    for field in dataclasses.fields(quant_config)
                ]
                raise ValueError(
                    f"Unrecognized parameter token: {token} in {param_tokens}\nAvailable parameters for {quant_config.__name__}: {field_info}"
                )

        return params
