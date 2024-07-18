# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Linear layer in tensor+sequence parallel with UB gemm overlap (tp-comm-overlap)"""

import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops

from utils import assert_allclose, assert_shape, set_random_seed
import transformer_engine.paddle as te

class _TestLinearTpBase(unittest.TestCase):
    """Tests Linear layer with column/row parallelism in BF16"""

    def setUp(self):
        self.set_attr()
        self.init_dist_env()
        paddle.set_default_dtype(self.global_dtype)

    def init_dist_env(self):
        """Init Paddle Fleet environment"""
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
        }
        strategy.hybrid_configs["mp_configs"].need_broadcast_data = False
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()
        self.hcg = fleet.get_hybrid_communicate_group()
        self.tp_group = self.hcg.get_model_parallel_group()
        self.world_size = self.hcg.get_model_parallel_world_size()

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.in_features = 32
        self.out_features = 64
        self.global_dtype = "bfloat16"
        self.rtol = 1e-3
        self.atol = 1e-3
        self.fp8 = False
        self.sequence_parallel = False

    def _train_one_step(self, layer, inp, optimizer, split_input="none", gather_output=False):
        inp = paddle.to_tensor(inp, stop_gradient=True)
        assert split_input in ["none", "column", "row"]
        if split_input == "column":
            split_size = inp.shape[1] // self.world_size
            input_parallel = inp[:, split_size * self.rank : split_size * (self.rank + 1)]
        elif split_input == "row":
            split_size = inp.shape[0] // self.world_size
            input_parallel = inp[split_size * self.rank : split_size * (self.rank + 1), :]
        else:
            input_parallel = inp
        if not input_parallel.is_contiguous():
            input_parallel = input_parallel.contiguous()
        input_parallel.stop_gradient = False
        assert input_parallel.is_contiguous(), 'Native TE kernel assume the buffer is all contiguous'
        out = layer(input_parallel)
        if gather_output:
            total_out = mp_ops._c_concat(out, group=self.tp_group)
        else:
            total_out = out
        loss = total_out.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if split_input != "none":
            grad_input = []
            paddle.distributed.all_gather(grad_input, input_parallel.grad, group=self.tp_group)
            if split_input == "column":
                grad_input = paddle.concat(grad_input, axis=1)
            elif split_input == "row":
                grad_input = paddle.concat(grad_input, axis=0)
        else:
            grad_input = input_parallel.grad
        return loss, grad_input
    
    def _create_pd_linear(self, layer_te: te.Linear, axis: int=0):
        """Create a normal Paddle nn.Linear with weight=[in_features, out_features] for comparing result"""
        layer_pd = te.Linear(
            self.in_features,
            self.out_features,
            backend="paddle",
        )
        # Get total weight
        total_weight = []
        partial_weight = layer_te.weight.clone().detach()
        paddle.distributed.all_gather(total_weight, partial_weight, group=self.tp_group)
        total_weight = paddle.concat(total_weight, axis=axis)
        layer_pd.weight.copy_(total_weight.T, True)
        return layer_pd

##############################################################################
# Unittest for Linear layer in tp-comm-overlap, 
# (which imply both tensor parallel + sequence parallel is applied as 'comm')
##############################################################################
class TestLinearUbOverlapRS(_TestLinearTpBase):
    """Tests Linear layer with column/row parallelism in BF16"""
    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.in_features = 64*4
        self.out_features = 64
        self.global_dtype = "bfloat16"
        self.rtol = 0.01
        self.atol = 0.001
        self.fp8 = False
        self.sequence_parallel = True
        
    def test_fc2_layer(self):
        """Tests fc2(row parallel linear) overlapping with RS(Reduce scatter)"""
        set_random_seed(1024)
        
        FFN = self.in_features
        H = self.out_features
        
        te.initialize_ub([self.batch_size, H], paddle.bfloat16, self.model_parallel_size)    
        
        layer_te = te.Linear(
            self.in_features,
            self.out_features,
            parallel_mode="row",
            sequence_parallel=self.sequence_parallel,
            ub_overlap_rs = True,
            ub_overlap_ag = True,
            ub_name=te.UbGEMM.fc2_fprop,
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


if __name__ == "__main__":
    unittest.main()
