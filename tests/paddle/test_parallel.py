# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE Paddle Parallel"""

from pathlib import Path
import unittest
import os
from unittest import mock

from dist_launcher import TestDistributed
from utils import is_devices_enough, is_multicast_supported

from transformer_engine.paddle.fp8 import is_fp8_available

test_root = Path(__file__).resolve().parent
gpu_has_fp8, reason = is_fp8_available()
gpu_has_multicast = is_multicast_supported()

class TestParallelLinear(TestDistributed):
    """Test Linear in Parallel mode"""

    @unittest.skipIf(not is_devices_enough(2), "TestParallelLinear needs 2 GPUs")
    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_linear_tp(self):
        """Tests linear with tensor parallel in BF16"""
        self.run_2gpu(str(test_root / "parallel_tests" / "linear_tp.py"))

    @unittest.skipIf(not is_devices_enough(2), "TestParallelLinear needs at least 2 GPUs")
    @mock.patch.dict(os.environ, {"UB_SKIPMC": "1"})
    def test_linear_tp_comm_overlap_ipc(self):
        """Tests GEMM+AG/GEMM+RS on te.Linear"""
        self.run_2gpu(str(test_root / "parallel_tests" / "linear_tp_comm_overlap.py"))
    @unittest.skipIf(not is_devices_enough(2), "TestParallelLinear needs at least 2 GPUs")
    @unittest.skipUnless(gpu_has_multicast, "No multicast supported")
    def test_linear_tp_comm_overlap(self):
        """Tests GEMM+AG/GEMM+RS on te.Linear"""
        self.run_2gpu(str(test_root / "parallel_tests" / "linear_tp_comm_overlap.py"))

class TestParallelLayerNormLinear(TestDistributed):
    """Test LayerNormLinear in Parallel mode"""

    @unittest.skipIf(not is_devices_enough(2), "TestParallelLayerNormLinear needs 2 GPUs")
    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_layernorm_linear_tp(self):
        """Tests layernorm_linear with tensor parallel in BF16"""
        self.run_2gpu(str(test_root / "parallel_tests" / "layernorm_linear_tp.py"))


class TestParallelLayerNormMLP(TestDistributed):
    """Test LayerNormMLP in Parallel mode"""

    @unittest.skipIf(not is_devices_enough(2), "TestParallelLayerNormMLP needs 2 GPUs")
    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_layernorm_mlp_tp(self):
        """Tests layernorm_mlp with tensor parallel in BF16"""
        self.run_2gpu(str(test_root / "parallel_tests" / "layernorm_mlp_tp.py"))

    @unittest.skipIf(not is_devices_enough(2), "TestParallelLayerNormMLP needs 2 GPUs")
    @mock.patch.dict(os.environ, {"UB_SKIPMC": "1"})
    def test_layernorm_mlp_tp_comm_overlap_ipc(self):
        """Tests te.LayerNormMLP when gemm is overlapped with AG+RS"""
        self.run_2gpu(str(test_root / "parallel_tests" / "layernorm_mlp_tp_comm_overlap.py"))
    @unittest.skipIf(not is_devices_enough(2), "TestParallelLayerNormMLP needs 2 GPUs")
    @unittest.skipUnless(gpu_has_multicast, "No multicast supported")
    def test_layernorm_mlp_tp_comm_overlap(self):
        """Tests te.LayerNormMLP when gemm is overlapped with AG+RS"""
        self.run_2gpu(str(test_root / "parallel_tests" / "layernorm_mlp_tp_comm_overlap.py"))

class TestAmaxReduction(TestDistributed):
    """Test amax reduction in dp mode"""

    @unittest.skipIf(not is_devices_enough(2), "TestAmaxReduction needs 2 GPUs")
    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_amax_reduction(self):
        """Tests amax reduction"""
        self.run_2gpu(str(test_root / "parallel_tests" / "amax_reduction.py"))


class TestPipelineParallel(TestDistributed):
    """Test pipeline parallel"""

    @unittest.skipIf(not is_devices_enough(2), "TestPipelineParallel needs 2 GPUs")
    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_pipeline_parallel(self):
        """Tests pipeline parallel"""
        self.run_2gpu(str(test_root / "parallel_tests" / "linear_pp.py"))


class TestGroupSharding(TestDistributed):
    """Test group sharding"""

    @unittest.skipIf(not is_devices_enough(2), "TestGroupSharding needs 2 GPUs")
    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_group_sharding(self):
        """Tests group sharding"""
        self.run_2gpu(str(test_root / "parallel_tests" / "group_sharding.py"))


class TestParallelAttention(TestDistributed):
    """Test MultiHeadAttention Layer in Parallel mode"""

    @unittest.skipIf(not is_devices_enough(2), "TestParallelAttention needs 2 GPUs")
    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_attention_tp(self):
        """Tests MultiHeadAttention Layer with tensor parallel in BF16"""
        self.run_2gpu(str(test_root / "parallel_tests" / "attention_tp.py"))

    @unittest.skipIf(not is_devices_enough(2), "TestParallelAttention needs 2 GPUs")
    @mock.patch.dict(os.environ, {"UB_SKIPMC": "1"})
    def test_attention_tp_comm_overlap_ipc(self):
        """Tests te.MultiHeadAttention when gemm is overlapped with AG+RS"""
        self.run_2gpu(str(test_root / "parallel_tests" / "attention_tp_comm_overlap.py"))
    @unittest.skipIf(not is_devices_enough(2), "TestParallelAttention needs 2 GPUs")
    @unittest.skipUnless(gpu_has_multicast, "No multicast supported")
    def test_attention_tp_comm_overlap(self):
        """Tests te.MultiHeadAttention when gemm is overlapped with AG+RS"""
        self.run_2gpu(str(test_root / "parallel_tests" / "attention_tp_comm_overlap.py"))


class TestParallelTransformerLayer(TestDistributed):
    """Test Transformer Layer in Parallel mode"""

    @unittest.skipIf(not is_devices_enough(2), "TestParallelTransformerLayer needs 2 GPUs")
    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_transformer_tp(self):
        """Tests Transformer Layer with tensor parallel in BF16"""
        self.run_2gpu(str(test_root / "parallel_tests" / "transformer_tp.py"))

    @unittest.skipIf(not is_devices_enough(2), "TestParallelTransformerLayer needs 2 GPUs")
    @mock.patch.dict(os.environ, {"UB_SKIPMC": "1"})
    def test_transformer_tp_comm_overlap_ipc(self):
        """Tests te.TransformerLayer when gemm is overlapped with AG+RS"""
        self.run_2gpu(str(test_root / "parallel_tests" / "transformer_tp_comm_overlap.py"))
    @unittest.skipIf(not is_devices_enough(2), "TestParallelTransformerLayer needs 2 GPUs")
    @unittest.skipUnless(gpu_has_multicast, "No multicast supported")
    def test_transformer_tp_comm_overlap(self):
        """Tests te.TransformerLayer when gemm is overlapped with AG+RS"""
        self.run_2gpu(str(test_root / "parallel_tests" / "transformer_tp_comm_overlap.py"))


if __name__ == "__main__":
    unittest.main()
