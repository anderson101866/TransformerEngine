# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Linear layer in tensor+sequence parallel with UB gemm overlap (tp-comm-overlap)"""

import unittest

import paddle
from paddle.distributed import fleet

from utils import assert_allclose, assert_shape, set_random_seed
from parallel_tests.linear_tp import _TestLinearTpBase
import transformer_engine.paddle as te

B = 16
H = 64

##############################################################################
# Unittest for Linear layer in tp-comm-overlap,
# (which imply both tensor parallel + sequence parallel is applied as 'comm')
##############################################################################
class TestLinearUbOverlapRS(_TestLinearTpBase):
    """Tests Linear layer with row parallelism in BF16"""
    def set_attr(self):
        """Set test configs"""
        self.batch_size = B
        self.in_features = H*4
        self.out_features = H
        self.global_dtype = "bfloat16"
        self.rtol = 0.01
        self.atol = 0.001
        self.fp8 = False
        self.sequence_parallel = True

    def test_fc2_layer(self):
        """Tests fc2(row parallel linear) overlapping with RS(Reduce scatter)"""
        set_random_seed(1024)

        FFN = self.in_features

        te.initialize_ub([self.batch_size, H], paddle.bfloat16, self.model_parallel_size)
        try:
            layer_te = te.Linear(
                self.in_features,
                self.out_features,
                parallel_mode="row",
                sequence_parallel=self.sequence_parallel,
                ub_overlap_rs = True,
                ub_overlap_ag = True,
                ub_name=te.UbGEMM.fc2,
            )

            layer_pd = self._create_pd_linear(layer_te, axis=1)

            assert_shape(
                layer_pd.weight, [FFN, H]
            )
            assert_shape(
                layer_te.weight, [H, FFN // self.model_parallel_size]
            )
            assert_shape(layer_te.bias, [H])

            optimizer_te = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_te.parameters())
            optimizer_pd = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_pd.parameters())

            layer_te = fleet.distributed_model(layer_te)
            optimizer_te = fleet.distributed_optimizer(optimizer_te)

            for _ in range(5):
                inp = paddle.rand([self.batch_size, FFN], self.global_dtype)

                loss_ref, grad_input_ref = self._train_one_step(layer_pd, inp, optimizer_pd)
                #with te.fp8_autocast(enabled=self.fp8):
                loss_tp, grad_input = self._train_one_step(
                    layer_te,
                    inp,
                    optimizer_te,
                    split_input="column",
                    gather_output=self.sequence_parallel,
                )
                assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
                assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)
        finally:
            te.destroy_ub()

if __name__ == "__main__":
    unittest.main()
