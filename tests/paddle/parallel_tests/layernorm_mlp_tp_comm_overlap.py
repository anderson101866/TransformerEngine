# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for LayerNormMLP layer in tensor+sequence parallel with UB gemm overlap (tp-comm-overlap)"""

import unittest

import paddle
from paddle.distributed import fleet

from utils import assert_allclose, assert_shape, set_random_seed
from parallel_tests.layernorm_mlp_tp import _TestLayerNormMLPTpBase
import transformer_engine.paddle as te

SEQUENCE_PARALLEL = True

##############################################################################
# Unittest for LayerNormMLP layer in tp-comm-overlap,
# (which imply both tensor parallel + sequence parallel is applied as 'comm')
##############################################################################
class TestLayerNormMLPTpCommOverlap(_TestLayerNormMLPTpBase):
    """Tests LayerNormMLP layer with model parallel in BF16"""
    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 32
        self.ffn_hidden_size = 64
        self.global_dtype = "bfloat16"
        self.rtol = 0.01
        self.atol = 0.001
        self.eps = 1e-3
        self.fp8 = False

    def test_parallel_layer(self):
        """Tests parallel LayerNormMLP with tp-comm-overlap enabled"""
        set_random_seed(1024)

        te.initialize_ub([self.batch_size, self.hidden_size], paddle.bfloat16, self.model_parallel_size)
        try:
            layer_te = te.LayerNormMLP(
                hidden_size=self.hidden_size,
                ffn_hidden_size=self.ffn_hidden_size,
                eps=self.eps,
                set_parallel_mode=True,
                sequence_parallel=SEQUENCE_PARALLEL,
                ub_overlap_rs=True,
                ub_overlap_ag=True,
            )
            layer_pd = self._create_ref_layer(layer_te)

            assert_shape(
                layer_te.fc1_weight,
                [self.ffn_hidden_size // self.model_parallel_size, self.hidden_size],
            )
            assert_shape(layer_te.fc1_bias, [self.ffn_hidden_size // self.model_parallel_size])
            assert_shape(
                layer_te.fc2_weight,
                [self.hidden_size, self.ffn_hidden_size // self.model_parallel_size],
            )
            assert_shape(layer_te.fc2_bias, [self.hidden_size])

            optimizer_te = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_te.parameters())
            optimizer_pd = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_pd.parameters())

            layer_te = fleet.distributed_model(layer_te)
            optimizer_te = fleet.distributed_optimizer(optimizer_te)

            for _ in range(5):
                inp = paddle.rand([self.batch_size, self.hidden_size], self.global_dtype)
                with te.fp8_autocast(enabled=self.fp8):
                    loss_tp, grad_input = self._train_one_step(
                        layer_te,
                        inp,
                        optimizer_te,
                        split_input="row" if SEQUENCE_PARALLEL else "none",
                        gather_output=SEQUENCE_PARALLEL,
                    )
                loss_ref, grad_input_ref = self._train_one_step(layer_pd, inp, optimizer_pd)
                assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
                assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)
        finally:
            te.destroy_ub()

if __name__ == "__main__":
    unittest.main()
