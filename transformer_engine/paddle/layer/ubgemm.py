# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Userbuffer related helper class/functions to configure how to perform tp-comm-overlap for nn Layers of TransformerEngine Paddle API"""
#NOTE: corresponding PyTorch helpers are located at "transformer_engine/pytorch/module/base.py"
from typing import Dict, Tuple, Union, Optional, Literal
from enum import Enum, auto
import os
import warnings

import socket
import fcntl
import struct

import paddle
from transformer_engine import transformer_engine_paddle as tex
from ..distributed import get_distributed_world_size, new_subgroups_by_enumeration
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

        assert (
            paddle.distributed.is_initialized()
        ), 'Have you run with "python -m paddle.distributed.launch"?' "paddle.distributed must be initialized before Userbuffers"
        bootstrap_backend = "nccl" #paddle only support NCCL for now

        world_group = paddle.distributed.new_group(backend=bootstrap_backend)
        world_rank = paddle.distributed.get_rank(world_group)
        world_size = get_distributed_world_size(world_group)

        # We have single-node NVLink so we can color based on physical node hostnames.
        # NOTE: If the user specified a valid network interface for NCCL or GLOO, use the host
        #       address on that interface instead of the hostname. Otherwise, allow the user to
        #       set a network interface via NVTE_UB_SOCKET_IFNAME variable. This can help avoid
        #       issues when  different hosts have the same hostname on managed clusters.
        mydomain = socket.gethostname()
        ifname = os.getenv(
            f"{bootstrap_backend.upper()}_SOCKET_IFNAME", os.getenv("NVTE_UB_SOCKET_IFNAME")
        )
        if ifname is not None:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                mydomain = socket.inet_ntoa(
                    fcntl.ioctl(
                        s.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode("UTF-8"))
                    )[20:24]
                )
            except OSError as err:
                raise OSError(f"Invalid network interface: {ifname}") from err
            finally:
                s.close()

        # Allgather the domain colors across ranks and reduce to a list of unique domains
        domain_per_rank_list = []
        paddle.distributed.all_gather_object(domain_per_rank_list, mydomain, world_group)
        unique_domains = []
        for domain in domain_per_rank_list:
            if domain not in unique_domains:
                unique_domains.append(domain)
        num_domains = len(unique_domains)

        if num_domains > 1:
            # DP/TP model replicated on multiple NVLink domains
            ranks_per_domain_list = [[] for _ in range(num_domains)]
            mydomain_idx = -1
            for i, domain in enumerate(domain_per_rank_list):
                domain_idx = unique_domains.index(domain)
                ranks_per_domain_list[domain_idx].append(i)
                if domain == mydomain:
                    mydomain_idx = domain_idx
            assert mydomain_idx >= 0, "Internal TE error!"

            intra_domain_group, _ = new_subgroups_by_enumeration(
                ranks_per_domain_list, backend=bootstrap_backend
            )
            local_rank = paddle.distributed.get_rank(intra_domain_group)
            local_size = paddle.distributed.get_world_size(intra_domain_group)

            inter_domain_group, _ = new_subgroups_by_enumeration(
                [list(ranks) for ranks in zip(*ranks_per_domain_list)],
                backend=bootstrap_backend,
            )
        else:
            # TP model on single NVLink domain, no replication, no data-parallelism
            mydomain_idx = 0
            local_rank = world_rank
            local_size = world_size
            intra_domain_group = world_group
            inter_domain_group = paddle.distributed.new_group([])

        node_id = world_rank // local_size
        num_nodes = world_size // local_size
        if world_rank == 0:
            print(f"!!! [UB] Number of NVLink domains: {num_domains}\n", end="", flush=True)
        if local_rank == 0:
            print(
                f"!!! [UB] Global ranks on domain {mydomain_idx}: {intra_domain_group}\n",
                end="",
                flush=True,
            )

        self.ub_pgs = {
            "world": world_group,        #define EXT_COMM_WORLD "world"
            "intra": intra_domain_group, #define EXT_COMM_INTRA "intra"
            "inter": inter_domain_group, #define EXT_COMM_INTER "inter"
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
        num_sm = 1
        cga_size = 1
        use_ce = True
        for ub_gemm in UbGEMM:
            for gemm_type in UbGemmType:
                sample_buffer = paddle.empty(shape, dtype=paddle.uint8 if use_fp8 and not is_reduce_scatter else dtype)
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
