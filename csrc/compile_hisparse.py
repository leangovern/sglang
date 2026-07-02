#!/usr/bin/env python3
"""
Compile HiSparse NPU Operator

Usage:
    python -m sglang.srt.hardware_backend.npu.csrc.compile_hisparse

This script compiles the Ascend C hisparse operator (hisparse_npu.cce)
into a .om file using the CANN ATC toolchain.

Requirements:
    - CANN toolkit installed and environment sourced
    - NPU device available (or at least ATC compiler available)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Default SOC versions for different NPU models
SOC_VERSIONS = {
    "910B": "Ascend910B",
    "910C": "Ascend910C",
    "950": "Ascend950",
    "310P": "Ascend310P",
}


def compile_hisparse_npu(
    cce_file: Path | None = None,
    output_dir: Path | None = None,
    soc_version: str = "Ascend910B",
    num_top_k: int = 2048,
    hot_buffer_size: int = 4096,
    item_size_bytes: int = 512,
) -> Path:
    """
    Compile the HiSparse NPU operator using ATC.

    Args:
        cce_file: Path to the .cce source file. If None, use default.
        output_dir: Directory to output the .om file. If None, use default.
        soc_version: NPU SOC version string for ATC.
        num_top_k: Number of top-k tokens (for naming).
        hot_buffer_size: Hot buffer size (for naming).
        item_size_bytes: Item size in bytes (for naming).

    Returns:
        Path to the generated .om file.
    """
    if cce_file is None:
        cce_file = Path(__file__).parent / "hisparse_npu.cce"

    if not cce_file.exists():
        raise FileNotFoundError(
            f"CCE source file not found: {cce_file}\n"
            f"Please ensure the source file exists before compiling."
        )

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "lib"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"hisparse_npu_{num_top_k}_{hot_buffer_size}_{item_size_bytes}"
    output_path = output_dir / f"{output_name}.om"

    # Build ATC command
    # Note: The exact ATC command may vary depending on CANN version.
    # Below are common variations. The script tries multiple approaches.

    # Approach 1: Using --singleop (CANN 7.0+)
    cmd1 = [
        "atc",
        "--singleop", str(cce_file),
        "--soc_version", soc_version,
        "--output", str(output_path),
        "--log=info",
    ]

    # Approach 2: Using --framework=5 (CANN 6.0+)
    cmd2 = [
        "atc",
        "--framework=5",
        "--model", str(cce_file),
        "--soc_version", soc_version,
        "--output", str(output_path.with_suffix("")),
        "--log=info",
    ]

    # Approach 3: Using op compilation helper (if available)
    cmd3 = [
        "op_compiler",
        "--cce_file", str(cce_file),
        "--soc_version", soc_version,
        "--output", str(output_path),
    ]

    commands = [
        ("ATC singleop", cmd1),
        ("ATC framework", cmd2),
        ("op_compiler", cmd3),
    ]

    for name, cmd in commands:
        print(f"\nTrying {name}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"\n✅ Compilation successful!")
            print(f"   Output: {output_path}")
            print(f"   SOC: {soc_version}")
            print(f"   Params: top_k={num_top_k}, buffer={hot_buffer_size}, item={item_size_bytes}")
            return output_path
        else:
            print(f"   Failed: {result.stderr[:500]}")

    raise RuntimeError(
        "All compilation approaches failed.\n"
        "Please ensure:\n"
        "  1. CANN is installed and environment is sourced (source set_env.sh)\n"
        "  2. atc/op_compiler is in your PATH\n"
        "  3. The .cce file has no syntax errors\n"
        "  4. Your SOC version is correct\n"
        f"Last error: {result.stderr[:1000]}"
    )


def compile_all_common_configs(
    output_dir: Path | None = None,
    soc_version: str = "Ascend910B",
) -> list[Path]:
    """
    Compile all common parameter configurations.

    Common configs for DeepSeek models:
        - top_k: 512, 1024, 2048
        - hot_buffer_size: 2048, 4096, 6144
        - item_size_bytes: 512 (MLA typical)
    """
    configs = [
        (512, 2048, 512),
        (512, 4096, 512),
        (1024, 2048, 512),
        (1024, 4096, 512),
        (1024, 6144, 512),
        (2048, 4096, 512),
        (2048, 6144, 512),
    ]

    outputs = []
    for num_top_k, hot_buffer_size, item_size_bytes in configs:
        print(f"\n{'='*60}")
        print(f"Compiling: top_k={num_top_k}, buffer={hot_buffer_size}, item={item_size_bytes}")
        print(f"{'='*60}")
        try:
            path = compile_hisparse_npu(
                output_dir=output_dir,
                soc_version=soc_version,
                num_top_k=num_top_k,
                hot_buffer_size=hot_buffer_size,
                item_size_bytes=item_size_bytes,
            )
            outputs.append(path)
        except Exception as e:
            print(f"   Warning: Failed to compile config {num_top_k}/{hot_buffer_size}/{item_size_bytes}: {e}")
            continue

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Compile HiSparse NPU Operator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile single configuration
  python compile_hisparse.py --soc_version Ascend910B --top_k 2048 --buffer 4096

  # Compile all common configurations
  python compile_hisparse.py --all

  # Compile for Ascend 950
  python compile_hisparse.py --all --soc_version Ascend950
        """,
    )
    parser.add_argument(
        "--soc_version",
        type=str,
        default="Ascend910B",
        choices=list(SOC_VERSIONS.values()) + list(SOC_VERSIONS.keys()),
        help="NPU SOC version (default: Ascend910B)",
    )
    parser.add_argument(
        "--top_k", type=int, default=2048, help="Number of top-k tokens"
    )
    parser.add_argument(
        "--buffer", type=int, default=4096, help="Hot buffer size"
    )
    parser.add_argument(
        "--item", type=int, default=512, help="Item size in bytes"
    )
    parser.add_argument(
        "--all", action="store_true", help="Compile all common configurations"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--cce_file", type=Path, default=None, help="Path to .cce source file"
    )

    args = parser.parse_args()

    # Normalize SOC version
    soc_version = SOC_VERSIONS.get(args.soc_version, args.soc_version)

    if args.all:
        outputs = compile_all_common_configs(
            output_dir=args.output_dir,
            soc_version=soc_version,
        )
        print(f"\n{'='*60}")
        print(f"Compiled {len(outputs)} configurations:")
        for path in outputs:
            print(f"  - {path}")
    else:
        output = compile_hisparse_npu(
            cce_file=args.cce_file,
            output_dir=args.output_dir,
            soc_version=soc_version,
            num_top_k=args.top_k,
            hot_buffer_size=args.buffer,
            item_size_bytes=args.item,
        )
        print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
