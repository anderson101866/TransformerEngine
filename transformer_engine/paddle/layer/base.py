# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Base modules and utilities for TransformerEngine Paddle API"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import os
import pickle
from typing import Generator, Dict, Tuple, Union, Any, List, Optional
from enum import Enum, auto
import socket

import numpy as np

import paddle

try:
    from paddle.base import core
    from paddle.base.framework import _dygraph_tracer
except ImportError:
    from paddle.fluid import core
    from paddle.fluid.framework import _dygraph_tracer

from transformer_engine import transformer_engine_paddle as tex

from ..constants import FP8FwdTensors, FP8BwdTensors, dist_group_type
from ..cpp_extensions import cast_transpose, cast_transpose_bgrad, cast_to_fp8, transpose
from ..fp8 import (
    FP8State,
    FP8TensorMeta,
    amax_and_scale_update,
    get_global_fp8_state,
    get_fp8_te_dtype,
)
from ..distributed import allgather, register_pp_fwd_begin_hook, is_pp_enabled, get_distributed_world_size
from ..profile import nvtx_range
from ..recompute import is_in_recompute_phase
from ..fp8_buffer import FP8RecomputeBuffer

__all__ = ["initialize_ub", "destroy_ub"]

_2X_ACC_FPROP = False
_2X_ACC_DGRAD = True
_2X_ACC_WGRAD = True
_cublas_workspace = None


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if paddle.device.cuda.get_device_capability()[0] >= 9:
        return 33_554_432
    return 4_194_304


def get_workspace() -> paddle.Tensor:
    """Returns workspace for cublas."""
    global _cublas_workspace
    if _cublas_workspace is None:
        _cublas_workspace = paddle.empty(
            [get_cublas_workspace_size_bytes()],
            dtype="uint8",
        )
    return _cublas_workspace

class UbGEMM(Enum):
    """GEMM layers that can apply tp-comm-overlap"""
    qkv_fprop  = auto()
    qkv_dgrad  = auto()
    proj_fprop = auto()
    proj_dgrad = auto()
    fc1_fprop  = auto()
    fc1_dgrad  = auto()
    fc2_fprop  = auto()
    fc2_dgrad  = auto()

    def with_reduce_scatter(self):
        """Return if `self` is a GEMM after RS operation in a Transformer block"""
        return self in {UbGEMM.proj_fprop, UbGEMM.fc2_fprop, UbGEMM.fc1_dgrad, UbGEMM.qkv_dgrad}

    def is_fprop(self):
        """Return if `self` is a forward propagation"""
        return self in _fprop_to_dgrad
    def get_dgrad(self):
        """Get the corresponding dgrad GEMM of the given forward propagation(`self`)"""
        return _fprop_to_dgrad[self]

    def is_qkv(self):
        """Return if this GEMM applies QKV projection"""
        return self in (UbGEMM.qkv_fprop, UbGEMM.qkv_dgrad)
    def is_proj(self):
        """Return if this GEMM applies output projection"""
        return self in (UbGEMM.proj_fprop, UbGEMM.proj_dgrad)

_fprop_to_dgrad = {
    UbGEMM.qkv_fprop:  UbGEMM.qkv_dgrad,
    UbGEMM.proj_fprop: UbGEMM.proj_dgrad,
    UbGEMM.fc1_fprop:  UbGEMM.fc1_dgrad,
    UbGEMM.fc2_fprop:  UbGEMM.fc2_dgrad,
}

_ub_manager = None
def initialize_ub(shape: Union[list, tuple], dtype, tp_size: int):
    """Initialize communicators for TP comm overlap using userbuffers."""
    global _ub_manager
    assert _ub_manager is None, "UB manager are already initialized."
    _ub_manager = _UBufGemmManager(shape, dtype, tp_size)

def get_ub(ub: UbGEMM) -> tex.CommGemmOverlapP2P:
    """Get userbuffer communicator corresponding to give key."""
    #NOTE: We don't implicitly expost this low-level API to user. Only te.Linear or other nn layer will use it.
    global _ub_manager
    assert _ub_manager is not None, "UB manager is not initialized."
    return _ub_manager.get_ub(ub)

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

        assert paddle.distributed.is_initialized()
        world_group = paddle.distributed.new_group(backend="nccl")
        world_rank = paddle.distributed.get_rank(world_group)
        world_size = get_distributed_world_size(world_group)

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
        global _cublas_workspace
        #_cublas_workspace = get_workspace().tile((tex.NVTE_COMM_OVERLAP_MAX_STREAMS,))
        _cublas_workspace = get_workspace().expand(shape=(tex.NVTE_COMM_OVERLAP_MAX_STREAMS, -1)).reshape((-1,))

        self.__set_bootstrap_callbacks()
        self.__add_ub(shape, dtype,
                      world_rank, world_size, local_rank, local_size, node_id, num_nodes, tp_size,
                      use_fp8)

    def get_ub(self, ub: UbGEMM):
        """Get userbuffer communicator corresponding to give key."""
        assert ub is not None, "TE internal error: nn Layers should ensure non-None `ub`, and reject user's bad argument"
        return self.__ub_communicators[ub]

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
        """preprate Ub object for each GEMM ops"""
        self.__ub_communicators = {}
        assert not use_fp8 and \
            dtype in {paddle.bfloat16, paddle.float16, paddle.float32, paddle.int32}, \
            "Currently, userbuffer comm-overlap doesn't support fp8"
        #P2P prefered options
        cga_size = 1
        use_ce = True
        for ub_gemm in UbGEMM:
            sample_buffer = paddle.empty(shape, dtype=paddle.uint8 if use_fp8 and not is_reduce_scatter else dtype)
            # Adjust SMs reserved for communication in MultiheadAttention
            if ub_gemm.is_qkv():
                num_sm = 8
            elif ub_gemm.is_proj():
                num_sm = 24
            else:
                num_sm = 4
            set_sm_margin = ub_gemm.with_reduce_scatter()
            is_reduce_scatter = ub_gemm.with_reduce_scatter()
            self.__ub_communicators[ub_gemm] = tex.CommGemmOverlapP2P(
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

class TransformerEngineBaseLayer(paddle.nn.Layer, ABC):
    """Base TE Layer."""

    def __init__(self) -> None:
        super().__init__()
        assert "gpu" in paddle.device.get_device(), "TransformerEngine needs CUDA."
        self.fp8_initialized = False
        self.fp8_enabled = False
        self.fp8_calibration = False
        self.fp8_meta = {}
        self.fp8_meta["fp8_checkpoint"] = False
        self.fp8_meta["fp8_group"] = None
        self.fp8_meta["recipe"] = FP8State.get_default_fp8_recipe()
        self.fp8_meta["scaling_fwd"] = FP8TensorMeta(is_forward=True)
        self.fp8_meta["scaling_bwd"] = FP8TensorMeta(is_forward=False)
        self.tp_group = None
        self.tp_size = 1
        self.sequence_parallel = False
        self.fp8_meta["autocast_id_fwd_stack"] = []
        self.fp8_meta["async_amax_reduction"] = bool(
            int(os.getenv("NVTE_ASYNC_AMAX_REDUCTION", "0"))
        )
        # weights that stored in fp16 would be cast into fp8 every first microstep
        self.fp8_weights = []
        self.fp8_weight_cache = {}
        self.registered_pp_start_callback = False

        self.current_step_id = paddle.to_tensor([1], dtype=paddle.int32, place=paddle.CPUPlace())

        def current_step_id_callback(step_id=None, **kwargs):  # pylint: disable=unused-argument
            self.current_step_id.copy_(
                paddle.to_tensor([step_id], dtype=paddle.int32, place=paddle.CPUPlace()), True
            )

        register_pp_fwd_begin_hook(current_step_id_callback)

    def set_activation_dtype(self, inp: paddle.Tensor) -> None:
        """Get activation data type for AMP."""
        tracer = _dygraph_tracer()
        if tracer and tracer._amp_level != core.AmpLevel.O0:
            # Set activation_dtype to the Paddle AMP dtype if under 'paddle.amp.auto_cast' context
            if tracer._amp_dtype == "float32":
                self.activation_dtype = paddle.float32
            elif tracer._amp_dtype == "bfloat16":
                self.activation_dtype = paddle.bfloat16
            elif tracer._amp_dtype == "float16":
                self.activation_dtype = paddle.float16
            else:
                raise RuntimeError(f"AMP format {tracer._amp_dtype} is not supported.")
        else:
            # If not under paddle.amp.auto_cast, set activation_dtype to the input dtype.
            # Also, make sure the parameters match the input dtype.

            # Skip the check if activation_dtype is already set and if activation_dtype
            # matches input dtype. If they do not match, e.g, when user switch from AMP
            # training to normal training, activation_dtype will still be updated.
            if hasattr(self, "activation_dtype") and self.activation_dtype == inp.dtype:
                return

            dtype = inp.dtype

            for name, param in self.named_parameters():
                if param is not None:
                    assert dtype == param.dtype, (
                        "Data types for parameters must match when outside of autocasted region. "
                        f" Found input dtype: {dtype} and {name!r} dtype: {param.dtype}"
                    )

            self.activation_dtype = dtype

    # This routine is shared across FP8 and FP8_calibration paths so should not actually
    # assume FP8 execution.
    def fp8_init(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        global_fp8_state = get_global_fp8_state()
        self.fp8_enabled = global_fp8_state.is_fp8_enabled()
        self.fp8_calibration = global_fp8_state.is_fp8_calibration()
        self.fp8_meta["fp8_checkpoint"] = self.fp8_enabled or self.fp8_calibration

        if self.fp8_enabled or self.fp8_calibration:
            # FP8 init has already been run and recipe is the same, don't do anything.
            if (
                self.fp8_initialized
                and global_fp8_state.get_fp8_recipe() == self.fp8_meta["recipe"]
            ):
                return

            # Set FP8, recipe, and other FP8 metadata
            self.fp8_meta["recipe"] = global_fp8_state.get_fp8_recipe()
            self.fp8_meta["fp8_group"] = global_fp8_state.get_fp8_group()

            # Set FP8_MAX per tensor according to recipe
            self.fp8_meta["fp8_max_fwd"] = self.fp8_meta["recipe"].fp8_format.value.max_fwd
            self.fp8_meta["fp8_max_bwd"] = self.fp8_meta["recipe"].fp8_format.value.max_bwd

            # Allocate scales and amaxes
            amax_history_len = self.fp8_meta["recipe"].amax_history_len
            self.fp8_meta["scaling_fwd"].prepare(num_gemms, amax_history_len)
            self.fp8_meta["scaling_bwd"].prepare(num_gemms, amax_history_len)
            self.fp8_initialized = True
        else:
            # If fp8 isn't enabled, turn off and return.
            self.fp8_initialized = False
            return

    def set_fp8_weights(self) -> None:
        """Initializes FP8 weights for the module"""
        if not self.fp8_enabled:
            return

        for i, weight in enumerate(self.fp8_weights, start=1):
            weight_cast_key = f"weight{i}_fp8"
            weight_transpose_key = f"weight{i}_t_fp8"

            if (
                weight_cast_key in self.fp8_weight_cache
                and self.fp8_weight_cache[weight_cast_key].shape == weight.shape
            ):
                return

            self.fp8_weight_cache[weight_cast_key] = paddle.empty(
                shape=weight.shape,
                dtype=paddle.uint8,
            )

            self.fp8_weight_cache[weight_transpose_key] = paddle.empty(
                shape=[weight.shape[1], weight.shape[0]],
                dtype=paddle.uint8,
            )

    def _get_fp8_state(self) -> paddle.Tensor:
        """Dump FP8 state to paddle.Tensor."""
        state = None
        if self.fp8_meta["fp8_checkpoint"]:
            state = {}
            state["scaling_fwd"] = self.fp8_meta["scaling_fwd"].to_numpy()
            state["scaling_bwd"] = self.fp8_meta["scaling_bwd"].to_numpy()
            state["global_fp8_fwd_buffer"] = get_global_fp8_state().get_fp8_fwd_buffer().to_numpy()
            state["global_fp8_bwd_buffer"] = get_global_fp8_state().get_fp8_bwd_buffer().to_numpy()
            # Store other pickelable values.
            extra = {}
            for k, v in self.fp8_meta.items():
                if isinstance(v, (bool, int, float, str)):
                    extra[k] = v
            state["extra_fp8_variables"] = extra

        state_serialized = pickle.dumps(state)
        state_tensor = paddle.to_tensor(np.frombuffer(state_serialized, dtype=np.uint8))

        return state_tensor

    @paddle.no_grad()
    def state_dict(
        self,
        destination=None,
        include_sublayers=True,
        structured_name_prefix="",
        use_hook=True,
    ):
        """Save FP8 State when checkpointing."""
        st = super().state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
            use_hook=use_hook,
        )
        st["fp8_state"] = self._get_fp8_state()
        return st

    def _set_fp8_state(self, state: paddle.Tensor) -> None:
        """Load previous state."""
        if state is None:
            return

        state = pickle.loads(state.numpy().tobytes())
        if state is None:
            return

        # Load fp8 meta tensors.
        self.fp8_meta["scaling_fwd"].from_numpy(state["scaling_fwd"])
        self.fp8_meta["scaling_bwd"].from_numpy(state["scaling_bwd"])

        # Restore global FP8 buffer states.
        global_fp8_fwd_buffer = get_global_fp8_state().get_fp8_fwd_buffer()
        global_fp8_bwd_buffer = get_global_fp8_state().get_fp8_bwd_buffer()
        global_fp8_fwd_buffer.from_numpy(state["global_fp8_fwd_buffer"])
        global_fp8_bwd_buffer.from_numpy(state["global_fp8_bwd_buffer"])

        # Load extra items.
        self.fp8_meta.update(state["extra_fp8_variables"])
        self.fp8_meta["recipe"].amax_history_len = self.fp8_meta["scaling_fwd"].amax_history.shape[
            0
        ]
        recompute_buffer_pos_key = FP8RecomputeBuffer.get_buffer_position_key()
        if recompute_buffer_pos_key in self.fp8_meta:
            del self.fp8_meta[recompute_buffer_pos_key]

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        """Restore FP8 State from checkpoint."""
        fp8_state_tensor = state_dict.pop("fp8_state")
        self._set_fp8_state(fp8_state_tensor)

        return super().set_state_dict(state_dict)

    @contextmanager
    def prepare_forward(
        self,
        inp: paddle.Tensor,
        is_first_microbatch: Union[bool, None],
        num_gemms: int = 1,
    ) -> Generator[paddle.Tensor, None, None]:
        """Checks and prep for FWD.
        The context manager is needed because there isn't a way for a module to know
        if it's the last FP8 module in the forward autocast. It is useful
        to setup the forward aggregated amax reduction for every module
        just in case. The autocast exit will pick up the most recent one.
        """

        if self.fp8_enabled and is_in_recompute_phase():
            global_recompute_buffer = get_global_fp8_state().get_fp8_recompute_buffer()
            global_recompute_buffer.retrieve_fp8_meta_tensors(self.fp8_meta)
        else:
            self.set_activation_dtype(inp)
            self.fp8_init(num_gemms=num_gemms)

            # Create persistent tensors for fp8 weights and their transposes
            # only when fp8 weight caching is used.
            if is_first_microbatch is not None:
                self.set_fp8_weights()

            if self.fp8_enabled and self.sequence_parallel:
                assert self.fp8_meta["recipe"].reduce_amax, (
                    "Amax reduction across tensor parallel group is "
                    "necessary when using sequence parallelism with FP8."
                )

            update_weight_scale_inv = is_first_microbatch is None or is_first_microbatch

            # Previous iteration was grad_enabled
            if self.fp8_meta.get("update_amax_and_scale_fwd", False):
                global_fp8_fwd_buffer = get_global_fp8_state().get_fp8_fwd_buffer()
                global_fp8_fwd_buffer.wait()
                if self.fp8_meta["recipe"].reduce_amax:
                    global_fp8_fwd_buffer.copy_amax_from_buffer(self.fp8_meta)
                    amax_and_scale_update(
                        self.fp8_meta,
                        fwd_update=True,
                        update_weight_scale_inv=update_weight_scale_inv,
                        current_step_id_tensor=self.current_step_id,
                        use_cudagraph=get_global_fp8_state().is_cudagraph_enabled(),
                    )
                    global_fp8_fwd_buffer.set_for_deletion(self.fp8_meta)
                else:
                    amax_and_scale_update(
                        self.fp8_meta,
                        fwd_update=True,
                        update_weight_scale_inv=update_weight_scale_inv,
                        current_step_id_tensor=self.current_step_id,
                        use_cudagraph=get_global_fp8_state().is_cudagraph_enabled(),
                    )

            if self.fp8_enabled and self.training:
                # Setup for amax reduction
                if self.fp8_meta["recipe"].reduce_amax:
                    global_fp8_state = get_global_fp8_state()
                    self.fp8_meta["first_module"] = global_fp8_state.is_first_fp8_module()
                    self.fp8_meta["autocast_id_fwd"] = global_fp8_state.get_autocast_id()
                    self.fp8_meta["autocast_id_fwd_stack"].append(self.fp8_meta["autocast_id_fwd"])
                self.fp8_meta["update_amax_and_scale_fwd"] = True
            else:
                self.fp8_meta["update_amax_and_scale_fwd"] = False

            # Activation recomputation is used and this is the first forward phase.
            if (
                self.fp8_enabled
                and self.training
                and get_global_fp8_state().is_fp8_recompute_enabled()
            ):
                global_recompute_buffer = get_global_fp8_state().get_fp8_recompute_buffer()
                global_recompute_buffer.stash_fp8_meta_tensors(self.fp8_meta)

        with nvtx_range(self.__class__.__name__ + " forward"):
            yield inp

        if self.fp8_enabled and is_in_recompute_phase():
            FP8RecomputeBuffer.restore_fp8_meta_tensors(self.fp8_meta)
            return

        if self.fp8_enabled and self.training and self.fp8_meta["recipe"].reduce_amax:
            global_fp8_state = get_global_fp8_state()
            global_fp8_fwd_buffer = global_fp8_state.get_fp8_fwd_buffer()
            global_fp8_fwd_buffer.add_amax(self.fp8_meta)
            global_fp8_fwd_buffer.set_for_amax_reduction(
                self.fp8_meta,
                self.tp_group,
                self.tp_size,
            )

    @staticmethod
    @contextmanager
    def prepare_backward(
        fp8_enabled: bool,
        fp8_meta: Dict[str, Any],
        tp_group: dist_group_type,
        tp_size: int,
        name: str = "",
    ) -> Generator[None, None, None]:
        """Checks and prep for BWD."""
        if fp8_enabled:
            global_fp8_state = get_global_fp8_state()
            global_fp8_bwd_buffer = global_fp8_state.get_fp8_bwd_buffer()
            global_fp8_bwd_buffer.wait()

            if fp8_meta["recipe"].reduce_amax:
                global_fp8_bwd_buffer.copy_amax_from_buffer(fp8_meta)
                amax_and_scale_update(
                    fp8_meta,
                    fwd_update=False,
                    use_cudagraph=get_global_fp8_state().is_cudagraph_enabled(),
                )
                global_fp8_bwd_buffer.set_for_deletion(fp8_meta)

                # Get new backward key.
                fp8_meta["autocast_id_bwd"] = fp8_meta["autocast_id_fwd_stack"].pop(0)
            else:
                amax_and_scale_update(
                    fp8_meta,
                    fwd_update=False,
                    use_cudagraph=get_global_fp8_state().is_cudagraph_enabled(),
                )

        with nvtx_range(name + " backward"):
            yield

        if fp8_enabled and fp8_meta["recipe"].reduce_amax:
            global_fp8_bwd_buffer.add_amax(fp8_meta)
            if fp8_meta["first_module"]:
                global_fp8_bwd_buffer.finalize(fp8_meta, tp_group, tp_size)

    @staticmethod
    def grad_output_preprocess(
        ctx, grad_output: paddle.Tensor, row_parallel_mode: bool
    ) -> Tuple[Union[paddle.Tensor, None], ...]:
        """Utility function for backward.
        Returns tuple in order (all optional/None based on training precion/recipe):
            R1: gathered `grad_output` in higher precision.
            R2: gathered `grad_output` in FP8.
            R3: R2 transposed.
            R4: bias gradient on R1.
        """
        grad_output_mat = grad_output.reshape((-1, grad_output.shape[-1]))
        gather_grad_output = row_parallel_mode and ctx.sequence_parallel

        # No-FP8 case: bgrad is fused with wgrad for this case.
        if not ctx.fp8_enabled:
            if gather_grad_output:
                if not ctx.ub_overlap_ag:
                    grad_output_mat, _ = allgather(grad_output_mat, ctx.tp_group)
                else:
                    ctx.ub_obj_gradout.copy_input_to_ubuf(grad_output, True) #do AG later with gemm
                    grad_output_mat = ctx.ub_obj_gradout.get_ubuf_output(
                        tex.NVTE_Comm_Overlap_Type.AG
                    )
            return grad_output_mat, None, None, None

        fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)

        if gather_grad_output:
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                # FP8 case with gather: unfused bgrad, cast, transpose for efficient gather
                if ctx.use_bias:
                    bgrad = grad_output_mat.sum(axis=0)
                else:
                    bgrad = None
                if ctx.ub_overlap_ag:
                    grad_output_c = ctx.ub_obj_gradout.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.RS)
                else:
                    grad_output_c = paddle.empty_like(grad_output_mat, dtype=paddle.uint8)
                #if not isinstance(grad_output_mat, Float8Tensor): #pytorch
                cast_to_fp8(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                    out=grad_output_c,
                )
                if not ctx.ub_overlap_ag:
                    grad_output_c, _ = allgather(grad_output_c, ctx.tp_group)
                    #if not isinstance(grad_output_c, Float8Tensor): #pytorch
                    grad_output_t = transpose(grad_output_c, fp8_dtype_backward)
                else:
                    grad_output_c = ctx.ub_obj_gradout.get_ubuf_output(tex.NVTE_Comm_Overlap_Type.AG)
                    grad_output_t = None

                return grad_output_mat, grad_output_c, grad_output_t, bgrad

            assert (
                not ctx.ub_overlap_ag
            ), "override_linear_precision.wgrad not supported with UB AG overlap"
            # FP8 case with gather and non-FP8 wgrad
            grad_output_mat, _ = allgather(grad_output_mat, ctx.tp_group)

        # FP8 case without gather: cast, transpose, bgrad fused
        if ctx.use_bias:
            bgrad, grad_output_c, grad_output_t = cast_transpose_bgrad(
                grad_output_mat,
                ctx.fp8_meta["scaling_bwd"],
                FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )
        else:
            if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                grad_output_c, grad_output_t = cast_transpose(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
            else:
                grad_output_t = None
                grad_output_c = cast_to_fp8(
                    grad_output_mat,
                    ctx.fp8_meta["scaling_bwd"],
                    FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )
            bgrad = None
        return grad_output_mat, grad_output_c, grad_output_t, bgrad

    @abstractmethod
    def forward(self):
        """Needs override."""

    def get_fp8_weights_scratchpad_and_cast(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[Optional[paddle.Tensor]]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`)
        """
        if not self.fp8_enabled or is_first_microbatch is None:
            return [None, None] * len(self.fp8_weights)

        out_list = []
        for i, _ in enumerate(self.fp8_weights, start=1):
            weight_cast_key = f"weight{i}_fp8"
            weight_transpose_key = f"weight{i}_t_fp8"

            assert (
                weight_cast_key in self.fp8_weight_cache
            ), "TE internal error: fp8 weight buffer is not found"

            weight_fp8 = self.fp8_weight_cache[weight_cast_key]
            weight_t_fp8 = self.fp8_weight_cache[weight_transpose_key]

            # Disable fp8 weight cache
            # is_first_microbatch is None -> we cast the weights into fp8 every micro step
            # Enalbe fp8 weight cache
            # is_first_microbatch == true -> we cast the weights into fp8 every micro step

            out_list.extend([weight_fp8, weight_t_fp8])

        # is cudagraph is enabled we cast the weight before the pp pipe
        # we only register the callback once
        if get_global_fp8_state().is_cudagraph_enabled() and (
            not self.registered_pp_start_callback and is_pp_enabled()
        ):

            fp8_dtype_forward = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=True)

            def cast_callback(step_id=None, **kwargs):  # pylint: disable=unused-argument
                update_fp8_weights = step_id == 0

                for i, weight in enumerate(self.fp8_weights, start=1):
                    weight_cast_key = f"weight{i}_fp8"
                    weight_transpose_key = f"weight{i}_t_fp8"

                    assert (
                        weight_cast_key in self.fp8_weight_cache
                    ), "TE internal error: fp8 weight buffer is not found"

                    weight_fp8 = self.fp8_weight_cache[weight_cast_key]
                    weight_t_fp8 = self.fp8_weight_cache[weight_transpose_key]

                    if paddle.is_grad_enabled():
                        if update_fp8_weights:
                            cast_transpose(
                                weight,
                                self.fp8_meta["scaling_fwd"],
                                (
                                    FP8FwdTensors.GEMM1_WEIGHT
                                    if i == 1
                                    else FP8FwdTensors.GEMM2_WEIGHT
                                ),
                                fp8_dtype_forward,
                                cast_out=weight_fp8,
                                transpose_out=weight_t_fp8,
                            )
                    else:
                        if update_fp8_weights:
                            cast_to_fp8(
                                weight,
                                self.fp8_meta["scaling_fwd"],
                                (
                                    FP8FwdTensors.GEMM1_WEIGHT
                                    if i == 1
                                    else FP8FwdTensors.GEMM2_WEIGHT
                                ),
                                fp8_dtype_forward,
                                out=weight_fp8,
                            )

            cast_callback(0 if is_first_microbatch else 1)
            register_pp_fwd_begin_hook(cast_callback)
            self.registered_pp_start_callback = True
        return out_list
