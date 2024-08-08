# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Transformer layer in tensor+sequence parallel with UB gemm overlap (tp-comm-overlap)"""

import unittest

import paddle
from paddle.distributed import fleet

from utils import assert_allclose, set_random_seed, register_sequence_parallel_allreduce_hooks
from parallel_tests.transformer_tp import _TestTransformerTpBase
import transformer_engine.paddle as te

SEQUENCE_PARALLEL = True

class TestTransformerTp(_TestTransformerTpBase):
    """Tests Transformer layer with model parallel in BF16"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 1024
        self.num_heads = 16
        self.ffn_hidden_size = 4096
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.mask_type = "padding"
        self.layer_type = "encoder"
        self.global_dtype = "bfloat16"
        self.rtol = 5e-2
        self.atol = 5e-2
        self.eps = 1e-3
        self.fp8 = False

    def test_parallel_layer(self):
        """Tests parallel Transformer"""
        set_random_seed(1024)
        common_args = [
            self.hidden_size,
            self.ffn_hidden_size,
            self.num_heads,
        ]
        common_kwargs = {
            "layernorm_epsilon": self.eps,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "self_attn_mask_type": self.mask_type,
            "layer_type": self.layer_type,
        }
        te.initialize_ub([self.batch_size*self.q_seqlen, self.hidden_size], paddle.bfloat16, self.model_parallel_size)
        try:
            layer_tp = te.TransformerLayer(
                *common_args,
                **common_kwargs,
                set_parallel_mode=True,
                sequence_parallel=SEQUENCE_PARALLEL,
                ub_tp_comm_overlap = True,
                ub_overlap_ag=True,
                ub_overlap_rs=True,
            )
            layer_single = self._create_ref_layer(layer_tp, common_args, common_kwargs)

            if SEQUENCE_PARALLEL:
                register_sequence_parallel_allreduce_hooks(layer_tp, accumulation_steps=1)

            optimizer_tp = paddle.optimizer.SGD(learning_rate=0.01, parameters=layer_tp.parameters())
            optimizer_single = paddle.optimizer.SGD(
                learning_rate=0.01, parameters=layer_single.parameters()
            )

            layer_tp = fleet.distributed_model(layer_tp)
            optimizer_tp = fleet.distributed_optimizer(optimizer_tp)

            for _ in range(5):
                inp = paddle.uniform(
                    [self.batch_size, self.q_seqlen, self.hidden_size], self.global_dtype
                )
                mask = paddle.zeros(
                    shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen), dtype="bool"
                )
                loss_tp, out_tp = self._train_one_step(
                    layer_tp, [inp, mask], optimizer_tp, self.fp8, SEQUENCE_PARALLEL
                )
                loss_single, out_single = self._train_one_step(
                    layer_single, [inp, mask], optimizer_single, self.fp8
                )
                assert_allclose(out_tp, out_single, rtol=self.rtol, atol=self.atol)
                assert_allclose(loss_tp, loss_single, rtol=self.rtol, atol=self.atol)
        finally:
            te.destroy_ub()


if __name__ == "__main__":
    unittest.main()
