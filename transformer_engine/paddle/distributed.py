# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Methods needed for distributed training."""

import os
import warnings
from contextlib import contextmanager
from typing import Any, Optional, Union, Tuple, List

import paddle

import paddle.distributed.fleet.base.topology as tp
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed import new_group, get_rank

try:
    # This feature is not supported as of Paddle 2.6.
    from paddle.distributed.fleet.meta_parallel import (
        PipelineParallelMicroStepLocations,
        register_global_pipeline_parallel_hook,
    )
except ImportError:
    print("Cannot find register_global_pipeline_parallel_hook !")
    register_global_pipeline_parallel_hook = None

from .constants import dist_group_type

_weight_split_axis = {
    "transformer_engine": {"row": 1, "column": 0},
    "paddle": {"row": 0, "column": 1},
}


def get_tp_group_and_world_size(
    tp_group: Union[dist_group_type, None], enable_tp: bool = True
) -> Tuple[Union[dist_group_type, None], int]:
    """Get TP group and world size using Fleet API"""
    if not (paddle.distributed.is_initialized() and enable_tp):
        return None, 1
    model_parallel_group = (
        tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group() if tp_group is None else tp_group
    )
    world_size = (
        tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
        if tp_group is None
        else tp_group.nranks
    )
    """
    When using TP, the NCCL communication needs to be scheduled
    before the GEMM for a guaranteed overlap. From the host side
    in TE, the comm calls are always launched first, but to ensure
    that the GEMM isn't scheduled first, the environment variable
    `CUDA_DEVICE_MAX_CONNECTIONS` needs to be set to 1 to force a
    single channel.
    """
    num_cuda_work_queues = int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0"))
    if num_cuda_work_queues != 1:
        warnings.warn(
            "To guarantee overlapping TP and SP collectives with the backward"
            "GEMMs, set environment variable CUDA_DEVICE_MAX_CONNECTIONS = 1"
        )

    return model_parallel_group, world_size

def get_distributed_world_size(group: Optional[dist_group_type] = None):
    """Get group size using paddle.distributed API or return 1 if paddle.distributed is not initialized"""
    if not paddle.distributed.is_initialized():
        return 1
    return paddle.distributed.get_world_size(group=group)

def is_pp_enabled() -> bool:
    """Check if pipeline parallel is enabled"""
    if not paddle.distributed.is_initialized():
        return False

    return tp._HYBRID_PARALLEL_GROUP.get_pipe_parallel_world_size() > 1


def register_pp_fwd_begin_hook(forward_begin_hook):
    """Register the pp hook if register_global_pipeline_parallel_hook exist"""
    if register_global_pipeline_parallel_hook is not None:
        register_global_pipeline_parallel_hook(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN, forward_begin_hook
        )


@contextmanager
def track_rng_state(enable: bool, **kwargs) -> None:
    """
    Applies get_rng_state_tracker().rng_state() to the context.
    If not enabled, it does nothing.
    """
    if enable:
        with get_rng_state_tracker().rng_state(**kwargs):
            yield
    else:
        yield


def set_tensor_dist_attr(tensor: paddle.Tensor, is_parallel: bool, axis: int) -> None:
    """Set distributed attributes for the input tensor"""
    tensor.is_distributed = is_parallel
    if is_parallel:
        tensor.split_axis = axis


def set_weight_tensor_dist_attr(
    tensor: paddle.Tensor, is_parallel: bool, parallel_mode: Optional[str], backend: str
) -> None:
    """Set distributed attributes for the weight tensor"""
    if not is_parallel or parallel_mode is None:
        return
    set_tensor_dist_attr(tensor, is_parallel, axis=_weight_split_axis[backend][parallel_mode])


def allreduce(
    input_: paddle.Tensor,
    tp_group: Optional[dist_group_type] = None,
    sync_op: bool = True,
) -> Tuple[paddle.Tensor, Any]:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if tp_group is None or tp_group.nranks == 1:
        return input_

    # All-reduce.
    if sync_op:
        output = mp_ops._mp_allreduce(
            input_,
            group=tp_group,
            use_calc_stream=True,
            use_model_parallel=True,
        )
        return output, None

    wait_handle = paddle.distributed.all_reduce(
        input_,
        op=paddle.distributed.ReduceOp.SUM,
        group=tp_group,
        sync_op=False,
    )

    output = input_

    return output, wait_handle


def allgather(
    input_: paddle.Tensor,
    tp_group: Optional[dist_group_type] = None,
    sync_op: bool = True,
) -> Tuple[paddle.Tensor, Any]:
    """All-gather the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if tp_group is None or tp_group.nranks == 1:
        return input_, None

    parallelism = tp_group.nranks
    output_shape = input_.shape
    output_shape[0] = output_shape[0] * parallelism
    output = paddle.empty(shape=output_shape, dtype=input_.dtype)
    wait_handle = tp_group.process_group.all_gather_into_tensor(output, input_, sync_op)
    if sync_op:
        wait_handle.wait()
        return output, None
    return output, wait_handle


def reduce_scatter(
    input_: paddle.Tensor,
    tp_group: Optional[dist_group_type] = None,
    sync_op: bool = True,
) -> [paddle.Tensor, Any]:
    """Reduce-scatter the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if tp_group is None or tp_group.nranks == 1:
        return input_, None

    parallelism = tp_group.nranks
    output_shape = input_.shape
    assert input_.shape[0] % parallelism == 0, (
        f"Input sequence length {input_.shape[0]} can't be divided "
        f"exactly by sequence parallelism {parallelism}"
    )
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input_.dtype)
    wait_handle = paddle.distributed.stream.reduce_scatter(
        output, input_, op=paddle.distributed.ReduceOp.SUM, group=tp_group, sync_op=sync_op
    )
    if sync_op:
        return output, None
    return output, wait_handle


def identity(
    input_: paddle.Tensor,
    tp_group: Optional[dist_group_type] = None,
) -> paddle.Tensor:
    """
    Identity when forward.
    Allreduce across model parallel group when backward.
    """
    output = mp_ops._c_identity(input_, group=tp_group)

    return output


def mark_as_sequence_parallel_parameter(parameter: paddle.Tensor):
    """
    Set sequence_parallel attribute to input tensor. It is used for registering allreduce
    hooks in PaddleNLP sequence parallel training.
    """
    setattr(parameter, "sequence_parallel", True)

def new_subgroups_by_enumeration( #reference to the implementation of PyTorch because Paddle 2.6.1 have no such API
                                  # see also: https://github.com/pytorch/pytorch/blob/d6d9183456cd07ca0b361a194b98c2fb196e7c36/torch/distributed/distributed_c10d.py#L4841
    ranks_per_subgroup_list: List[List[int]],
    timeout=None,
    backend=None,
) -> Tuple['Group', List['Group']]:
    """
    Create subgroups by dividing the global world.

    The division is specified by a nested list of ranks. The subgroups cannot have
    overlap, and some ranks may not have to be in any subgroup.

    This is a convenience API that calls ``new_group`` to generate multiple subgroups.
    It requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group.

    .. warning::
        See warning `Safe concurrent usage` for `new_group` API for important details about
        using multiple process groups concurrently in a safe manner.

    Args:
        ranks_per_subgroup_list (list[list[int]]): A nested list of ranks of
            group members.
        timeout (datetime.timedelta, optional): see `new_group` for details and default value.

    Returns:
        The subgroup containing the current rank, and all the subgroups used for cleanup.

    Examples:
        >>> # Create two subgroups, where each has 2 processes.
        >>> cur_subgroup, subgroups = new_subgroups_by_enumeration([[0, 2], [1, 3]])
        >>> rank = dist.get_rank()
        >>> tensor = paddle.ones(1).cuda() * rank
        >>> dist.all_reduce(tensor, group=cur_subgroup)
        >>> tensor
        tensor([2])     # Subgroup 0: ranks 0 and 2
        tensor([4])     # Subgroup 1: ranks 1 and 3
    """
    if ranks_per_subgroup_list is None or len(ranks_per_subgroup_list) == 0:
        raise ValueError("The arg 'ranks_per_subgroup_list' cannot be empty")

    subgroups = []
    cur_subgroup = None
    # Create a mapping from rank to subgroup to check if there is any subgroup overlap.
    rank_to_ranks_dict = {}  # type: ignore[var-annotated]
    for ranks in ranks_per_subgroup_list:
        subgroup = new_group(
            ranks=ranks,
            timeout=timeout,
            backend=backend,
        )
        subgroups.append(subgroup)
        my_rank = get_rank()
        for rank in ranks:
            if rank in rank_to_ranks_dict:
                raise ValueError(
                    f"Rank {rank} has appeared in both subgroup {rank_to_ranks_dict[rank]} and {ranks}"
                )
            rank_to_ranks_dict[rank] = ranks
            if my_rank == rank:
                cur_subgroup = subgroup
                #logger.info("Rank %s is assigned to subgroup %s", rank, ranks)

    return cur_subgroup, subgroups
