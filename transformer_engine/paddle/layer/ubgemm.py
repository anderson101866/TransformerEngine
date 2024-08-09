# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Userbuffer related helper class/functions to configure how to perform tp-comm-overlap for nn Layers of TransformerEngine Paddle API"""
#NOTE: corresponding PyTorch helpers are located at "transformer_engine/pytorch/module/base.py"
from typing import Dict, Tuple, Union, Optional, Literal
from enum import Enum, auto
import os
import socket
import warnings
import paddle

from transformer_engine import transformer_engine_paddle as tex
from ..distributed import get_distributed_world_size
from .base import expand_workspace

__all__ = ["initialize_ub", "destroy_ub"]


class UbGEMM(Enum):
    """
    In a transformer layer, enums for GEMM layers that can apply tp-comm-overlap
    """
    qkv  = auto()
    proj = auto()
    fc1  = auto()
    fc2  = auto()

    @property
    def can_column_parallel(self) -> bool:
        """Return if this GEMM supports column-parallel"""
        return self in (UbGEMM.fc1, UbGEMM.qkv)
    @property
    def can_row_parallel(self) -> bool:
        """Return if this GEMM supports row-parallel"""
        return self in (UbGEMM.fc2, UbGEMM.proj)

    @staticmethod
    def parse(ub_name: str) -> 'UbGEMM':
        """Parse the user given string into `UbGEMM` for TE implementation. Raise ValueError if parsing fail"""
        try:
            return UbGEMM[ub_name]
        except KeyError:
            raise ValueError(f"Invalid ub_name:'{ub_name}'. Must be one of: {', '.join(UbGEMM.__members__.keys())}") # pylint: disable=raise-missing-from

class UbGemmType(Enum):
    """Different kinds of `UbGEMM` in a transformer layer, which is used to generate a combinations of any (`UbGemm`, `UbGemmType`) pairs"""
    fprop = auto()
    dgrad = auto()

_ub_manager = None
def initialize_ub(shape: Union[list, tuple], dtype, tp_size: int):
    """Initialize communicators for TP comm overlap using userbuffers."""
    global _ub_manager
    assert _ub_manager is None, "UB manager are already initialized."
    _ub_manager = _UBufGemmManager(shape, dtype, tp_size)

def get_ub(gemm: UbGEMM, gemm_type: UbGemmType) -> tex.CommGemmOverlapP2P:
    """Get userbuffer communicator corresponding to give key."""
    #NOTE: We don't implicitly expose this low-level API to user. Only te.Linear or other nn layer will use it.
    global _ub_manager
    assert _ub_manager is not None, "UB manager is not initialized."
    return _ub_manager.get_ub(gemm, gemm_type)

def destroy_ub():
    """Destroy all allocated userbuffer communicators."""
    global _ub_manager
    _ub_manager = None

class _UBufGemmManager: #pylint: disable=too-few-public-methods
    def __init__(
        self,
        shape: Union[list, tuple],
        dtype: paddle.dtype,
        tp_size: int,
        use_fp8: bool = False,
    ):
        """
        Args:
            shape: the to stored a batch of data sample. e.g. [SxB, H]
        """
        assert len(shape) == 2, 'shape should be [SxB, H]'

        if not tex.device_supports_multicast():
            assert (
                bool(os.getenv("UB_SKIPMC", None))
            ), (
                "CUDA device, driver and/or toolkit version does not support comm+GEMM overlap with "
                + "CUDA Multicast. Launch app with UB_SKIPMC=1 to try CUDA IPC instead."
            )

        assert paddle.distributed.is_initialized(), 'Have you run with "python -m paddle.distributed.launch"?'
        world_group = paddle.distributed.new_group(backend="nccl")
        world_rank = paddle.distributed.get_rank(world_group)
        world_size = get_distributed_world_size(world_group)

        assert shape[0] % tp_size == 0, (
            f"Given shape [SxB, H]={shape}, SxB({shape[0]}) can't be divided "
            f"exactly by sequence parallelism {tp_size}"
        )

        # Construct an intra-node communicator -- this should include ALL ranks in the node
        # NOTE: This may be different than the tensor-parallel group (e.g. two TP groups in a node),
        #       in which case the local_size we get below will not be equal to the tp_size given
        #       by the user. Userbuffers internally accounts for this.
        hostnames = [None for _ in range(world_size)]
        hostname = socket.gethostname()
        paddle.distributed.all_gather_object(hostnames, hostname)
        intra_node_ranks = []
        for i, host in enumerate(hostnames):
            if host == hostname:
                intra_node_ranks.append(i)
        if len(intra_node_ranks) == world_size:
            intra_node_group = world_group
            local_rank = world_rank
            local_size = world_size
        else:
            intra_node_group = paddle.distributed.new_group(backend="nccl", ranks=intra_node_ranks)
            local_rank = paddle.distributed.get_rank(intra_node_group)
            local_size = paddle.distributed.get_world_size(intra_node_group)

        node_id = world_rank // local_size
        num_nodes = world_size // local_size
        if local_rank == 0:
            print(
                f"Found {num_nodes} physical node{'s' if num_nodes > 1 else ''}\n"
                + f"Global ranks on node {node_id}: {intra_node_ranks}\n",
                end='',
                flush=True
            )

        self.ub_pgs = {
            "world": world_group,      #static char EXT_COMM_WORLD[] = "world";
            "intra": intra_node_group, #static char EXT_COMM_INTRA[] = "intra";
        }

        # Increase the workspace by the number of maximum concurrent streams
        expand_workspace(tex.NVTE_COMM_OVERLAP_MAX_STREAMS)

        self.__set_bootstrap_callbacks()
        self.__add_ub(shape, dtype,
                      world_rank, world_size, local_rank, local_size, node_id, num_nodes, tp_size,
                      use_fp8)

    def get_ub(self, ub: UbGEMM, gemm_type: UbGemmType):
        """Get userbuffer communicator corresponding to give key."""
        assert ub is not None, "TE internal error: nn Layers should ensure non-None `ub`, and reject user's bad argument"
        return self.__ub_communicators[(ub, gemm_type)]

    def __set_bootstrap_callbacks(self):
        """Set the collective API provided by paddle framework, to implement TP comm overlap using userbuffers."""
        def allgather_callback(global_data: paddle.Tensor, local_data: paddle.Tensor, group: str):
            assert (
                global_data.place.is_cpu_place() and local_data.place.is_cpu_place()
            ), ("TE internal error: Comm+GEMM overlap bootstrap callbacks need host (CPU) tensors."
              f" global_data:{global_data.place} local_data:{local_data.place}")

            # Move tensors to device if using NCCL backend
            pg = self.ub_pgs[group]
            if paddle.distributed.get_backend(pg) == "NCCL":
                gathered_data_in_gpu = paddle.empty_like(global_data).cuda()
                paddle.distributed.all_gather(gathered_data_in_gpu, local_data.cuda(), group=pg)
                # Copy global tensor from CUDA back to original CPU tensor
                paddle.assign(gathered_data_in_gpu.cpu(), output=global_data)
                assert global_data.place.is_cpu_place(), f"TE internal error: Bootstrap callbacks need fill data into host (CPU) tensors, but not at {global_data.place}"
            else:
                paddle.distributed.all_gather(global_data, local_data, group=pg)

        def bcast_callback(data: paddle.Tensor, src: int, group: str):
            # Move tensor to device if using NCCL backend
            assert (
                data.place.is_cpu_place()
            ), "TE internal error: Comm+GEMM overlap bootstrap callbacks need host (CPU) tensors."

            pg = self.ub_pgs[group]
            if paddle.distributed.get_backend(pg) == "NCCL":
                data_in_gpu = data.cuda()
                paddle.distributed.broadcast(data_in_gpu, src, pg)

                # Copy global tensor from CUDA back to original CPU tensor and clear temporary tensor
                paddle.assign(data_in_gpu.cpu(), output=data)
            else:
                paddle.distributed.broadcast(data, src, pg)

        def barrier_callback(group: str):
            paddle.distributed.barrier(group=self.ub_pgs[group])

        tex.set_comm_overlap_callbacks(tex._dist_callback_holder, allgather_callback, bcast_callback, barrier_callback)

    def __add_ub(self, shape, dtype,
                 world_rank, world_size, local_rank, local_size, node_id, num_nodes, tp_size,
                 use_fp8):
        """prepare Ub object for each GEMM ops"""
        self.__ub_communicators: Dict[Tuple[UbGEMM, UbGemmType], tex.CommGemmOverlapP2P] = {}
        assert not use_fp8 and \
            dtype in {paddle.bfloat16, paddle.float16, paddle.float32, paddle.int32}, \
            "Currently, userbuffer comm-overlap doesn't support fp8"
        #P2P preferred options
        cga_size = 1
        use_ce = True
        for ub_gemm in UbGEMM:
            for gemm_type in UbGemmType:
                sample_buffer = paddle.empty(shape, dtype=paddle.uint8 if use_fp8 and not is_reduce_scatter else dtype)
                # Adjust SMs reserved for communication in MultiheadAttention
                if ub_gemm == UbGEMM.qkv:
                    num_sm = 8
                elif ub_gemm == UbGEMM.proj:
                    num_sm = 24
                else:
                    num_sm = 4
                is_reduce_scatter = (gemm_type == UbGemmType.fprop and ub_gemm.can_row_parallel) or \
                                    (gemm_type == UbGemmType.dgrad and ub_gemm.can_column_parallel)
                set_sm_margin = is_reduce_scatter
                self.__ub_communicators[(ub_gemm, gemm_type)] = tex.CommGemmOverlapP2P(
                    sample_buffer,  # Sample userbuffer
                    world_rank,  # Global rank
                    world_size,  # Number of global ranks
                    local_rank,  # Local rank in physical node
                    local_size,  # Number of local ranks in physical node
                    node_id,  # Physical node ID
                    num_nodes,  # Number of physical nodes
                    tp_size,  # Tensor-parallel group size (may be smaller than local_size)
                    tex.NVTE_COMM_OVERLAP_MAX_STREAMS,  # Max. number of compute streams (default: 3)
                    cga_size,  # CGA cluster size
                    num_sm,  # Number of communication SMs
                    set_sm_margin,  # Set SM margin
                    use_ce,  # Use copy engine
                    False,   # Use a single GEMM with atomic-counters
                    False,
                    is_reduce_scatter,  # Overlapped collective is reduce-scatter
                )

def validate_ub_args(parallel_mode: Literal["row", "column", None], backend: str, ub_overlap_rs: bool, ub_overlap_ag: bool, ub_name: Optional[str]) -> Optional[UbGEMM]:
    """A helper function to reject meaningless argument for nn.Layer like `te.Linear`, `te.LayerNormLinear`"""
    if backend == "paddle" and (ub_overlap_rs or ub_overlap_ag or ub_name):
        warnings.warn(
            "userbuffer overlapping (tp-comm-overlap) is not supported for paddle backend and all `ub_*` arguments will be ignored."
        )
    if ub_overlap_rs or ub_overlap_ag:
        assert ub_name is not None, "Userbuffer name (`ub_name`) is not set."
        ub_name: UbGEMM = UbGEMM.parse(ub_name)
        if parallel_mode == "row":
            assert ub_name.can_row_parallel, f"The given ub_name:`{ub_name.name}` doesn't support row-parallel linear"
        elif parallel_mode == "column":
            assert ub_name.can_column_parallel, f"The given ub_name:`{ub_name.name}` doesn't support column-parallel linear"
    elif ub_name is not None:
        warnings.warn("Please set `ub_overlap_rs` or `ub_overlap_ag` to enable userbuffer overlapping (tp-comm-overlap), or `ub_name` argument is ignored.")
        ub_name = None
    return ub_name
