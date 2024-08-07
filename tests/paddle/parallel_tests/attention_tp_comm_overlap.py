# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Transformer layer in tensor parallel"""

import unittest

import paddle
from paddle.distributed import fleet

from utils import assert_allclose, set_random_seed, register_sequence_parallel_allreduce_hooks
from parallel_tests.attention_tp import _TestAttentionTpBase
import transformer_engine.paddle as te

SEQUENCE_PARALLEL = True

class TestAttentionTpCommOverlap(_TestAttentionTpBase):
    """Tests MultiHeadAttention layer with model parallel in BF16"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 1024
        self.num_heads = 16
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.mask_type = "padding"
        self.global_dtype = "bfloat16"
        self.rtol = 5e-3
        self.atol = 5e-3
        self.eps = 1e-3
        self.fp8 = False

    def test_parallel_layer(self):
        """Tests parallel Transformer"""
        set_random_seed(1024)
        common_args = (
            self.hidden_size,
            self.num_heads,
        )
        common_kwargs = {
            "layernorm_epsilon": self.eps,
            "attention_dropout": 0.0,
            "attn_mask_type": self.mask_type,
            "attention_type": "self",
            "tp_group": self.tp_group,
            "input_layernorm": True,
        }
        te.initialize_ub([self.batch_size*self.q_seqlen, self.hidden_size], paddle.bfloat16, self.model_parallel_size)
        try:
            layer_tp = te.MultiHeadAttention(
                *common_args,
                **common_kwargs,
                set_parallel_mode=True,
                sequence_parallel=SEQUENCE_PARALLEL,
                ub_overlap_rs=True,
                ub_overlap_ag=True,
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
