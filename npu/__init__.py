# NPU hardware backend module for SGlang
# This module contains NPU-specific implementations of various operators
# and backends.

from sglang.srt.hardware_backend.npu.hisparse_npu import (
    load_cache_to_device_buffer_mla,
    load_cache_to_device_buffer_dsv4_mla,
)

__all__ = [
    "load_cache_to_device_buffer_mla",
    "load_cache_to_device_buffer_dsv4_mla",
]
