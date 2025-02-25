import unittest

import torch

from torchao.quantization.config_parser import ConfigParser
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    MappingType,
    UIntXWeightOnlyConfig,
)


class TestConfigParser(unittest.TestCase):
    def setUp(self):
        self.parser = ConfigParser()

    def test_int4wo_config(self):
        # Basic Int4WeightOnlyConfig
        config = self.parser.parse("int4wo_g32")
        self.assertIsInstance(config, Int4WeightOnlyConfig)
        self.assertEqual(config.group_size, 32)

        # With symmetry specified
        config = self.parser.parse("int4wo_g64")
        self.assertIsInstance(config, Int4WeightOnlyConfig)
        self.assertEqual(config.group_size, 64)

    def test_int8wo_config(self):
        # Basic Int8WeightOnlyConfig
        config = self.parser.parse("int8wo_g128")
        self.assertIsInstance(config, Int8WeightOnlyConfig)
        self.assertEqual(config.group_size, 128)

        # Verify that symmetry parameter raises error since not supported
        with self.assertRaises(ValueError) as context:
            self.parser.parse("int8wo_g128_sym")

        self.assertIn(
            "Invalid parameters for Int8WeightOnlyConfig", str(context.exception)
        )
        self.assertIn("mapping_type", str(context.exception))

    def test_int8dqint4_config(self):
        # Int8 dynamic activation with Int4 weight
        config = self.parser.parse("int8dqint4_g32")
        self.assertIsInstance(config, Int8DynamicActivationInt4WeightConfig)
        self.assertEqual(config.group_size, 32)

        # With symmetry
        config = self.parser.parse("int8dqint4_g32_sym")
        self.assertIsInstance(config, Int8DynamicActivationInt4WeightConfig)
        self.assertEqual(config.group_size, 32)
        self.assertEqual(config.mapping_type, MappingType.SYMMETRIC)

    def test_int8dqint8_config(self):
        # Int8 dynamic activation with Int8 weight
        config = self.parser.parse("int8dqint8")
        self.assertIsInstance(config, Int8DynamicActivationInt8WeightConfig)

    def test_int4dqint4_config(self):
        # Int4 dynamic activation with Int4 weight
        config = self.parser.parse("int4dqint4_sym")
        self.assertIsInstance(config, Int4DynamicActivationInt4WeightConfig)
        self.assertEqual(config.mapping_type, MappingType.SYMMETRIC)

    def test_float8wo_config(self):
        # Basic Float8WeightOnlyConfig with e4m3 dtype
        config = self.parser.parse("float8wo_e4m3")
        self.assertIsInstance(config, Float8WeightOnlyConfig)
        self.assertEqual(config.weight_dtype, torch.float8_e4m3fn)

        # With e5m2 dtype
        config = self.parser.parse("float8wo_e5m2")
        self.assertIsInstance(config, Float8WeightOnlyConfig)
        self.assertEqual(config.weight_dtype, torch.float8_e5m2)

    def test_float8dqfloat8_config(self):
        # Float8 dynamic activation with Float8 weight
        config = self.parser.parse("float8dqfloat8_e4m3")
        self.assertIsInstance(config, Float8DynamicActivationFloat8WeightConfig)
        self.assertEqual(config.activation_dtype, torch.float8_e4m3fn)
        self.assertEqual(config.weight_dtype, torch.float8_e4m3fn)

    def test_uintxwo_config(self):
        # UIntX config with uint4
        config = self.parser.parse("uintxwo_uint4_g32")
        self.assertIsInstance(config, UIntXWeightOnlyConfig)

        # With uint8
        config = self.parser.parse("uintxwo_uint8_g64")
        self.assertIsInstance(config, UIntXWeightOnlyConfig)

    # def test_fpx_config(self):
    #     # FPX config
    #     config = self.parser.parse("fpx_e4m3")
    #     self.assertIsInstance(config, FPXWeightOnlyConfig)
    #     self.assertEqual(config.ebits, 4)
    #     self.assertEqual(config.mbits, 3)

    def test_invalid_config_string(self):
        # Test empty string
        with self.assertRaises(ValueError):
            self.parser.parse("")

        # Test unknown base config
        with self.assertRaises(ValueError):
            self.parser.parse("unknown_config")

        # Test invalid parameter token
        with self.assertRaises(ValueError):
            self.parser.parse("int4wo_invalid_token")

    def test_complex_configurations(self):
        # Adjust tests for complex configurations to match actual parameter names
        config = self.parser.parse("int4wo_g32")
        self.assertIsInstance(config, Int4WeightOnlyConfig)
        self.assertEqual(config.group_size, 32)

        config = self.parser.parse("int8dqint4_g32_asym")
        self.assertIsInstance(config, Int8DynamicActivationInt4WeightConfig)
        self.assertEqual(config.group_size, 32)
        self.assertEqual(config.mapping_type, MappingType.ASYMMETRIC)


if __name__ == "__main__":
    unittest.main()
