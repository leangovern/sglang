from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sglang.srt.utils.common import is_npu

if is_npu():
    import torch_npu

logger = logging.getLogger(__name__)

# Path to the precompiled NPU operator library
NPU_OP_LIB_DIR = Path(__file__).parent / "lib"


@functools.lru_cache(maxsize=128)
def _get_npu_hisparse_module(
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
) -> int:
    """
    Load the precompiled NPU hisparse operator (.om file).

    Returns the model_id from aclmdlLoadFromFile.
    The module is cached by (item_size_bytes, num_top_k, hot_buffer_size).
    """
    if not is_npu():
        raise RuntimeError("NPU hisparse operator can only be loaded on NPU devices")

    import acl

    om_filename = f"hisparse_npu_{num_top_k}_{hot_buffer_size}_{item_size_bytes}.om"
    om_path = NPU_OP_LIB_DIR / om_filename

    if not om_path.exists():
        # Try the generic fallback name (if compiled without parameter-specific naming)
        om_path_fallback = NPU_OP_LIB_DIR / "hisparse_npu.om"
        if om_path_fallback.exists():
            om_path = om_path_fallback
        else:
            raise FileNotFoundError(
                f"NPU hisparse operator not found: {om_path}\n"
                f"Please compile it first:\n"
                f"  1. Ensure CANN environment is set up (source set_env.sh)\n"
                f"  2. Run: python -m sglang.srt.hardware_backend.npu.csrc.compile_hisparse\n"
                f"  3. Or manually: atc --singleop=... --output={om_path}\n"
                f"See: sglang/srt/hardware_backend/npu/csrc/hisparse_npu.cce"
            )

    # Load the .om model using ACL
    model_id, ret = acl.mdl.load_from_file(str(om_path))
    if ret != 0:
        raise RuntimeError(
            f"Failed to load NPU hisparse operator from {om_path}, "
            f"ACL error code: {ret}"
        )

    logger.info(
        f"Loaded NPU hisparse operator: {om_path.name} "
        f"(model_id={model_id}, top_k={num_top_k}, buffer={hot_buffer_size}, item={item_size_bytes})"
    )
    return model_id


def _create_acl_dataset(
    tensors: list[torch.Tensor],
    dataset_type: str,  # "input" or "output"
) -> int:
    """
    Create an ACL dataset from a list of PyTorch tensors.

    Args:
        tensors: List of torch tensors.
        dataset_type: "input" or "output" for logging.

    Returns:
        aclmdlDataset handle (int).
    """
    import acl

    dataset = acl.mdl.create_dataset()
    for idx, tensor in enumerate(tensors):
        # Get tensor data pointer and size
        data_ptr = tensor.data_ptr()
        data_size = tensor.numel() * tensor.element_size()

        # Create data buffer and add to dataset
        data_buffer = acl.create_data_buffer(data_ptr, data_size)
        ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
        if ret != 0:
            raise RuntimeError(
                f"Failed to add {dataset_type} buffer {idx} to ACL dataset, "
                f"error code: {ret}"
            )

    return dataset


def _execute_npu_op(
    model_id: int,
    input_tensors: list[torch.Tensor],
    output_tensors: list[torch.Tensor],
    stream,
) -> None:
    """
    Execute a precompiled NPU operator using ACL.

    Args:
        model_id: Loaded model ID from aclmdlLoadFromFile.
        input_tensors: List of input torch tensors.
        output_tensors: List of output torch tensors (pre-allocated).
        stream: NPU stream (torch_npu.npu.Stream).
    """
    import acl

    # Create input and output datasets
    input_dataset = _create_acl_dataset(input_tensors, "input")
    output_dataset = _create_acl_dataset(output_tensors, "output")

    try:
        # Execute the model
        # If stream is a torch_npu stream, get its ACL stream handle
        if hasattr(stream, "npu_stream"):
            acl_stream = stream.npu_stream
        elif hasattr(stream, "_npu_stream"):
            acl_stream = stream._npu_stream
        else:
            # Fallback: use default stream
            acl_stream = 0

        ret = acl.mdl.execute_async(model_id, input_dataset, output_dataset, acl_stream)
        if ret != 0:
            raise RuntimeError(f"ACL model execution failed, error code: {ret}")

        # Synchronize to ensure completion
        # In production, this should be async; here we sync for correctness
        if acl_stream != 0:
            ret = acl.rt.synchronize_stream(acl_stream)
        else:
            ret = acl.rt.synchronize_device()

        if ret != 0:
            raise RuntimeError(f"ACL stream synchronization failed, error code: {ret}")

    finally:
        # Cleanup datasets
        # Note: ACL buffers inside the dataset are owned by the dataset
        # and should be destroyed with the dataset
        acl.mdl.destroy_dataset(input_dataset)
        acl.mdl.destroy_dataset(output_dataset)


# ---------------------------------------------------------------------------
# Public API: matches the CUDA JIT kernel interface exactly
# ---------------------------------------------------------------------------

def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    """
    NPU implementation of load_cache_to_device_buffer_mla for HiSparse.

    This is the functional equivalent of the CUDA JIT-compiled kernel,
    using a precompiled Ascend C operator (.om file).

    Args:
        top_k_tokens: [num_reqs, num_top_k] int32 - selected token positions
        device_buffer_tokens: [num_reqs, hot_buffer_size] int32 - cached tokens
        host_cache_locs: [num_reqs, max_seq_len] int64 - host cache indices
        device_buffer_locs: [num_reqs, hot_buffer_size] int32 - device buffer indices
        host_cache: Host KV cache data (uint8/byte tensor)
        device_buffer: Device KV buffer data (uint8/byte tensor)
        top_k_device_locs: output [num_reqs, num_top_k] int32 - device locations
        req_pool_indices: [num_reqs] int32/int64 - request pool indices
        seq_lens: [num_reqs] int32/int64 - sequence lengths
        lru_slots: [num_reqs, hot_buffer_size] uint32 - LRU timestamps
            NOTE: NPU uses uint32 timestamps instead of int16 physical ordering.
            The Python caller must ensure lru_slots is uint32.
        item_size_bytes: size of each KV item in bytes
        num_top_k: number of top-k tokens
        hot_buffer_size: size of device hot buffer
        page_size: page size (unused in NPU, kept for compatibility)
        block_size: CUDA block size (unused in NPU, kept for compatibility)
        num_real_reqs: [1] int32 - actual number of requests (for CUDA graph safety)
    """
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    # Ensure tensors are on NPU device
    device = top_k_tokens.device
    if device.type != "npu":
        raise ValueError(f"NPU hisparse requires NPU device, got {device}")

    # Handle num_real_reqs
    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=device
        )

    # Validate lru_slots dtype: must be uint32 for NPU
    if lru_slots.dtype != torch.uint32:
        # If caller passes int16 (CUDA format), convert to uint32 on first use
        # This is a compatibility shim for mixed CUDA/NPU code paths
        if lru_slots.dtype == torch.int16:
            logger.warning(
                "lru_slots is int16 (CUDA format), converting to uint32 for NPU. "
                "For best performance, initialize lru_slots as uint32 on NPU."
            )
            # Convert: int16 physical ordering -> uint32 timestamps
            # For newly initialized slots, assign sequential timestamps
            lru_slots = lru_slots.to(torch.int32)
            # Create timestamps based on position (higher = more recent)
            batch_size = lru_slots.size(0)
            timestamps = (
                torch.arange(hot_buffer_size, dtype=torch.uint32, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            lru_slots = timestamps
        else:
            lru_slots = lru_slots.to(torch.uint32)

    # Load the precompiled NPU operator
    model_id = _get_npu_hisparse_module(
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
    )

    # Prepare input tensors in the order expected by the .cce operator:
    # 0: top_k_tokens
    # 1: device_buffer_tokens
    # 2: host_cache_locs
    # 3: device_buffer_locs
    # 4: host_cache_k
    # 5: device_buffer_k
    # 6: lru_slots (uint32)
    # 7: num_real_reqs
    # Plus scalar params via additional tensor or custom mechanism
    #
    # NOTE: The .cce operator expects these 8 tensor inputs + 4 scalar params
    # (num_reqs, item_size_bytes, num_top_k, hot_buffer_size).
    # In ACL, scalar params are typically passed as additional 1-element tensors.

    num_reqs_val = top_k_tokens.size(0)

    input_tensors = [
        top_k_tokens,                          # 0
        device_buffer_tokens,                  # 1
        host_cache_locs,                       # 2
        device_buffer_locs,                    # 3
        host_cache,                            # 4
        device_buffer,                         # 5
        lru_slots,                             # 6
        num_real_reqs,                         # 7
        # Scalar params as 1-element tensors on NPU
        torch.tensor([num_reqs_val], dtype=torch.int32, device=device),      # 8
        torch.tensor([item_size_bytes], dtype=torch.int64, device=device),   # 9
        torch.tensor([num_top_k], dtype=torch.int32, device=device),         # 10
        torch.tensor([hot_buffer_size], dtype=torch.int32, device=device),   # 11
    ]

    output_tensors = [top_k_device_locs]

    # Get current NPU stream
    stream = torch_npu.npu.current_stream()

    # Execute the operator
    _execute_npu_op(model_id, input_tensors, output_tensors, stream)


def load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    """
    DSv4 variant of NPU hisparse load cache.

    Currently falls back to the generic MLA implementation.
    TODO: Add DSv4-specific page-padded C4 layout support.
    """
    logger.warning(
        "DSv4 hisparse on NPU currently falls back to generic MLA path. "
        "DSv4 C4 page-padded layout support is planned."
    )
    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
    )
