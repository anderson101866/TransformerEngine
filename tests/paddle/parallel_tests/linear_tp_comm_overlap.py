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
SEQUENCE_PARALLEL = True

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

    def test_fc2_layer(self):
        """Tests fc2(row-parallel linear) overlapping with RS(Reduce scatter)"""
        set_random_seed(1024)

        FFN = self.in_features

        te.initialize_ub([self.batch_size, H], paddle.bfloat16, self.model_parallel_size)
        try:
            layer_te = te.Linear(
                self.in_features,
                self.out_features,
                parallel_mode="row",
                sequence_parallel=SEQUENCE_PARALLEL,
                ub_overlap_rs = True,
                ub_overlap_ag = True,
                ub_name='fc2',
            )

            layer_pd = self._create_ref_layer(layer_te, axis=1)

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
                    gather_output=SEQUENCE_PARALLEL,
                )
                assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
                assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)
        finally:
            te.destroy_ub()

class TestLinearUbOverlapAG(_TestLinearTpBase):
    """Tests Linear layer with column parallelism in BF16"""
    def set_attr(self):
        """Set test configs"""
        self.batch_size = B
        self.in_features = H
        self.out_features = H*4
        self.global_dtype = "bfloat16"
        self.rtol = 0.01
        self.atol = 0.001
        self.fp8 = False

    def test_fc1_layer(self):
        """Tests fc1(column-parallel linear) overlapping with AG(allgather)"""
        set_random_seed(1024)

        FFN = self.out_features

        te.initialize_ub([self.batch_size, H], paddle.bfloat16, self.model_parallel_size)
        try:
            layer_te = te.Linear(
                self.in_features,
                self.out_features,
                parallel_mode="column",
                sequence_parallel=SEQUENCE_PARALLEL,
                ub_overlap_rs = True,
                ub_overlap_ag = True,
                ub_name='fc1',
            )

            layer_pd = self._create_ref_layer(layer_te, axis=0)

            assert_shape(
                layer_pd.weight, [H, FFN]
            )
            assert_shape(
                layer_te.weight, [FFN // self.model_parallel_size, H]
            )
            assert_shape(layer_te.bias, [FFN // self.model_parallel_size])

            optimizer_te = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_te.parameters())
            optimizer_pd = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_pd.parameters())

            layer_te = fleet.distributed_model(layer_te)
            optimizer_te = fleet.distributed_optimizer(optimizer_te)

            for _ in range(5):
                inp = paddle.rand([self.batch_size, H], self.global_dtype)

                loss_ref, grad_input_ref = self._train_one_step(layer_pd, inp, optimizer_pd)
                #with te.fp8_autocast(enabled=self.fp8):
                loss_tp, grad_input = self._train_one_step(
                    layer_te,
                    inp,
                    optimizer_te,
                    split_input="row",
                    gather_output=SEQUENCE_PARALLEL,
                )
                assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
                assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)
        finally:
            te.destroy_ub()

if __name__ == "__main__":
    unittest.main()
