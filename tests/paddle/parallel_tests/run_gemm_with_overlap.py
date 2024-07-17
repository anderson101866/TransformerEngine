#!/usr/bin/python3

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
from functools import partial
import paddle
from paddle.distributed import fleet
import transformer_engine.paddle as te
from transformer_engine import transformer_engine_paddle as tex
from transformer_engine.paddle.cpp_extensions import (
    gemm,
)

nvte_comm_types = {
    "ag": tex.NVTE_Comm_Overlap_Type.AG,
    "rs": tex.NVTE_Comm_Overlap_Type.RS,
}

def mapped_argtype(opt, typemap={}):
    if str(opt).lower() not in typemap.keys():
        raise TypeError(f"Unrecognized option! Please choose from: {typemap.keys()}")
    return typemap[str(opt).lower()]

def parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Test comm+GEMM overlap with Userbuffers.")
    parser.add_argument(
        "--timing-iters",
        type=int,
        default=1,
        help="Benchmark the comm+GEMM overlap as an average of many iterations.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument("--hidden-size", type=int, default=64*128)
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--comm-type",
        type=partial(mapped_argtype, typemap=nvte_comm_types),
        default=tex.NVTE_Comm_Overlap_Type.AG,
        help="Comm type to overlap.",
    )
    
    opts = parser.parse_args(argv, namespace)
    return opts

def train(args, tp_group):
    tp_size = paddle.distributed.get_world_size(tp_group)
    
    # Initialize userbuffers with (M, N) buffer
    # M = sequence * batch
    # N = hidden size
    outer_size = args.batch_size*args.seq_length
    te.initialize_ub((outer_size, args.hidden_size), paddle.bfloat16, tp_size)    
    
    ffn_hidden_size = 4 * args.hidden_size
    
    total_iter = args.timing_iters
    # Figure out problem sizing:
    # M = sequence * batch
    # N = hidden size
    # K = MLP intermediate size, named ffn_hidden_size (usually 4x hidden size)
    # P = number of devices for sequence/tensor parallelism
    print(f'M={outer_size}, N=H={args.hidden_size}, K={ffn_hidden_size}, P=TP=SP={tp_size}')
    
    if args.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
        # (M/P, N) -> overlapped AG -> (M, N) x (K/P, N)^T = (M, K/P)
        local_kernel_t_shape = (ffn_hidden_size // tp_size, args.hidden_size) #(K/P, N)
        local_inp_shape = (outer_size // tp_size, args.hidden_size)           #(M/P, N)
    else:
        # (M, K/P) x (N, K/P)^T = (M, N) -> overlapped RS -> (M/P, N)
        local_kernel_t_shape = (args.hidden_size, ffn_hidden_size // tp_size) #(N, K/P)
        local_inp_shape = (outer_size, ffn_hidden_size // tp_size)            #(M, K/P)
    
    kernel_t = paddle.rand(local_kernel_t_shape, dtype=paddle.bfloat16).cuda()
    inp = paddle.rand(local_inp_shape, dtype=paddle.bfloat16).cuda()
    
    # Gather global tensors and calculate reference result (need these first for Fp8 scales)
    if args.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
        # AG Kernel: (K/P, N) -> gather -> (K, N) -> T -> (N, K)        
        ker_g = paddle.transpose(te.distributed.allgather(kernel_t, tp_group)[0], (1, 0))
        # AG Input: (M/P, N) -> gather -> (M, N)
        inp_g = te.distributed.allgather(inp, tp_group)[0]
    else:
        # RS Kernel: (N, K/P) -> T -> (K/P, N) -> gather -> (K, N)
        ker_g = te.distributed.allgather(paddle.transpose(kernel_t, (1, 0)), tp_group)[0]
        # RS Input: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
        inp_g = paddle.transpose(
            te.distributed.allgather(paddle.transpose(inp, (1, 0)), tp_group)[0], (1, 0)
        )
    ref_g = paddle.matmul(inp_g, ker_g)
    assert args.fp8 == False, 'Not implemented yet'
    #TODO: Fp8 scales
    # ... ub_obj.set_ubuf_scale_inv
    
    # Set up comm/compute buffers
    if args.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
        ub_obj = te.get_ub(te.UbGEMM.fc1_fprop)
        ub_algo = tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_AG_P2P
        
        ub_obj.copy_input_to_ubuf(inp, True)
        gemm_inp = ub_obj.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.AG)
        ubuf_out = None
        rs_out = paddle.empty_like(ub_obj.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.RS))
    else:
        ub_obj = te.get_ub(te.UbGEMM.fc2_fprop)
        ub_algo = tex.NVTE_Comm_Overlap_Algo.SPLIT_PIPELINED_RS_P2P
        
        gemm_inp = inp
        ubuf_out = ub_obj.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.AG)
        rs_out = paddle.empty(
            (outer_size // tp_size, args.hidden_size), dtype=paddle.bfloat16)
    assert not kernel_t.place.is_cpu_place(), f'{kernel_t.place}'
    assert not gemm_inp.place.is_cpu_place(), f'{gemm_inp.place}'
    assert rs_out is None or not rs_out.place.is_cpu_place()  , f'{rs_out.place}'
    assert ubuf_out is None or not ubuf_out.place.is_cpu_place(), f'{ubuf_out.place}'
    for _ in range(total_iter):
        all_outputs = gemm(
            kernel_t,
            gemm_inp,
            paddle.bfloat16,
            te.layer.base.get_workspace(),
            bias=None,
            use_bias=False,
            gelu=False,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=rs_out,
            out=ubuf_out,
        )

    if args.comm_type == tex.NVTE_Comm_Overlap_Type.AG:
        # AG Output: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
        output = all_outputs[0]
        test_out = paddle.transpose(
            te.distributed.allgather(paddle.transpose(output, (1, 0)), tp_group),
            (1, 0),
        )
    else:
        # RS Output: (M/P, N) -> gather -> (M, N)
        output = rs_out
        test_out = te.distributed.allgather(output, tp_group)[0]

    # Compare against standard GEMM
    assert ref_g.shape == test_out.shape, f'ref_g.shape={ref_g.shape}, test_out.shape={test_out.shape}'
    rtol=0.125  if args.fp8 else 0.01
    atol=0.0675 if args.fp8 else 0.001
    assert paddle.allclose(ref_g.to(dtype=paddle.float32), test_out.to(dtype=paddle.float32), rtol=rtol, atol=atol), \
        f'paddle.isclose(ref_g, test_out, rtol={rtol}, atol={atol})={paddle.isclose(ref_g, test_out, rtol=rtol, atol=atol)}'
    
def main(args):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": paddle.distributed.get_world_size(),
        "pp_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    
    paddle.seed(args.seed)
    print(f'set seed={args.seed}')

    train(args, tp_group)
    te.destroy_ub()

if __name__ == "__main__":    
    args = parse_args()
    main(args)