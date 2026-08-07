"""Measure the PyNvVideoCodec feeder path one stage at a time.

The default suite compares raw NVDEC, RGB conversion, the exact DLPack +
torch.stack path used by pipeline_core.py, and ThreadedDecoder prefetching.
No frames are copied to the CPU and no output file is written.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import PyNvVideoCodec as nvc
import torch


@dataclass(frozen=True)
class Case:
    name: str
    decoder: str
    color: str
    consumer: str
    buffer_size: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NVDEC, PyNv RGB conversion, DLPack and torch.stack separately."
    )
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument("-b", "--batch-size", type=int, default=19)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the recommended comparison (default if no individual case is selected).",
    )
    parser.add_argument("--decoder", choices=("simple", "threaded"))
    parser.add_argument("--color", choices=("native", "rgb", "rgbp"), default="rgb")
    parser.add_argument("--consumer", choices=("none", "dlpack", "stack"), default="stack")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=0,
        help="ThreadedDecoder frame buffer; 0 means 2 * batch size.",
    )
    args = parser.parse_args()
    if args.frames <= 0 or args.batch_size <= 0:
        parser.error("--frames and --batch-size must be positive")
    if args.buffer_size < 0:
        parser.error("--buffer-size cannot be negative")
    if args.decoder is None:
        args.suite = True
    return args


def make_decoder(case: Case, source: str, gpu_id: int):
    color = getattr(nvc.OutputColorType, case.color.upper())
    common = dict(
        gpu_id=gpu_id,
        use_device_memory=True,
        output_color_type=color,
    )
    if case.decoder == "threaded":
        return nvc.ThreadedDecoder(source, case.buffer_size, **common)
    return nvc.SimpleDecoder(source, **common)


def close_decoder(decoder, kind: str) -> None:
    if kind == "threaded":
        decoder.end()
    # PyNvVideoCodec 2.2.0's Python SimpleDecoder.stop() wrapper calls a
    # non-existent native stop() method. Dropping the object closes it safely.


def run_case(case: Case, args: argparse.Namespace) -> dict[str, float | int | str]:
    init_start = time.perf_counter()
    decoder = make_decoder(case, str(args.input), args.gpu_id)
    init_seconds = time.perf_counter() - init_start
    available = len(decoder)
    target = min(args.frames, available)

    decoded = 0
    fetch_seconds = 0.0
    dlpack_seconds = 0.0
    stack_seconds = 0.0
    batches = 0

    torch.cuda.synchronize(args.gpu_id)
    wall_start = time.perf_counter()
    try:
        while decoded < target:
            request = min(args.batch_size, target - decoded)

            started = time.perf_counter()
            frames = decoder.get_batch_frames(request)
            fetch_seconds += time.perf_counter() - started
            if not frames:
                break

            tensors = None
            batch = None
            if case.consumer in ("dlpack", "stack"):
                started = time.perf_counter()
                tensors = [torch.from_dlpack(frame) for frame in frames]
                if case.consumer == "dlpack":
                    torch.cuda.synchronize(args.gpu_id)
                dlpack_seconds += time.perf_counter() - started

            if case.consumer == "stack":
                started = time.perf_counter()
                batch = torch.stack(tensors)
                # CUDA work is asynchronous. Synchronizing makes this the real
                # cost paid by the downstream pipeline, not submission time.
                torch.cuda.synchronize(args.gpu_id)
                stack_seconds += time.perf_counter() - started

            decoded += len(frames)
            batches += 1
            del batch, tensors, frames
    finally:
        torch.cuda.synchronize(args.gpu_id)
        wall_seconds = time.perf_counter() - wall_start
        close_decoder(decoder, case.decoder)
        del decoder
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "name": case.name,
        "frames": decoded,
        "batches": batches,
        "init": init_seconds,
        "fetch": fetch_seconds,
        "dlpack": dlpack_seconds,
        "stack": stack_seconds,
        "wall": wall_seconds,
        "fps": decoded / wall_seconds if wall_seconds else 0.0,
    }


def print_result(result: dict[str, float | int | str]) -> None:
    frames = int(result["frames"])
    print(f"\n--- {result['name']} ---")
    print(f"Frames / batches: {frames} / {result['batches']}")
    print(f"Init:             {result['init']:.3f} s")
    print(f"Fetch:            {result['fetch']:.3f} s  ({1000 * result['fetch'] / frames:.3f} ms/frame)")
    print(f"DLPack:           {result['dlpack']:.3f} s  ({1000 * result['dlpack'] / frames:.3f} ms/frame)")
    print(f"torch.stack:      {result['stack']:.3f} s  ({1000 * result['stack'] / frames:.3f} ms/frame)")
    print(f"Wall:             {result['wall']:.3f} s")
    print(f"Throughput:       {result['fps']:.2f} FPS")


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    buffer_2x = args.buffer_size or args.batch_size * 2
    buffer_4x = max(buffer_2x, args.batch_size * 4)
    if args.suite:
        cases = [
            Case("Simple / NATIVE / fetch only", "simple", "native", "none"),
            Case("Simple / RGB / fetch only", "simple", "rgb", "none"),
            Case("Simple / RGB / DLPack + stack", "simple", "rgb", "stack"),
            Case(f"Threaded / RGB / stack / buffer={buffer_2x}", "threaded", "rgb", "stack", buffer_2x),
            Case(f"Threaded / RGB / stack / buffer={buffer_4x}", "threaded", "rgb", "stack", buffer_4x),
        ]
    else:
        buffer_size = buffer_2x if args.decoder == "threaded" else 0
        cases = [Case("Custom", args.decoder, args.color, args.consumer, buffer_size)]

    print(f"PyNvVideoCodec: {nvc.__version__}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"GPU:            {torch.cuda.get_device_name(args.gpu_id)}")
    print(f"Input:          {args.input}")
    print(f"Target frames:  {args.frames}")
    print(f"Batch size:     {args.batch_size}")

    results = []
    for case in cases:
        result = run_case(case, args)
        results.append(result)
        print_result(result)

    print("\n===== SUMMARY =====")
    for result in results:
        print(f"{result['fps']:9.2f} FPS  {result['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
