# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for LayerNormLinear layer in tensor+sequence parallel with UB gemm overlap (tp-comm-overlap)"""

import unittest

import paddle
from paddle.distributed import fleet

from utils import assert_allclose, assert_shape, set_random_seed
from parallel_tests.layernorm_linear_tp import _TestLayerNormLinearTpBase
import transformer_engine.paddle as te

B = 16
H = 64
SEQUENCE_PARALLEL = True

##############################################################################
# Unittest for LayerNormLinear layer in tp-comm-overlap,
# (which imply both tensor parallel + sequence parallel is applied as 'comm')
##############################################################################
class TestLayerNormLinearTpCommOverlap(_TestLayerNormLinearTpBase):
    """Tests LayerNormLinear layer with column parallelism in BF16"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = B
        self.in_features = H
        self.out_features = H*4
        self.global_dtype = "bfloat16"
        self.rtol = 0.01
        self.atol = 0.001
        self.eps = 1e-3
        self.fp8 = False

    def test_column_parallel_layer(self):
        """Tests column parallel LayerNormLinear"""
        set_random_seed(1024)
        te.initialize_ub([self.batch_size, H], paddle.bfloat16, self.model_parallel_size)
        try:
            layer_te = te.LayerNormLinear(
                self.in_features,
                self.out_features,
                eps=self.eps,
                parallel_mode="column",
                sequence_parallel=SEQUENCE_PARALLEL,
                ub_overlap_ag=True,
                ub_name='qkv',
            )
            layer_pd = self._create_ref_layer(layer_te)

            assert_shape(
                layer_te.weight, [self.out_features // self.model_parallel_size, self.in_features]
            )
            assert_shape(layer_te.bias, [self.out_features // self.model_parallel_size])

            optimizer_te = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_te.parameters())
            optimizer_pd = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_pd.parameters())

            layer_te = fleet.distributed_model(layer_te)
            optimizer_te = fleet.distributed_optimizer(optimizer_te)

            for _ in range(5):
                inp = paddle.uniform([self.batch_size, self.in_features], self.global_dtype)
                with te.fp8_autocast(enabled=self.fp8):
                    loss_tp, grad_input = self._train_one_step(
                        layer_te,
                        inp,
                        optimizer_te,
                        split_input="row" if SEQUENCE_PARALLEL else "none",
                        gather_output=True,
                    )
                loss_ref, grad_input_ref = self._train_one_step(layer_pd, inp, optimizer_pd)
                assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
                assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)
        finally:
            te.destroy_ub()

if __name__ == "__main__":
    unittest.main()
