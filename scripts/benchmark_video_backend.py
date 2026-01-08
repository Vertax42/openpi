"""
Benchmark script to compare video decoding backend performance (pyav vs torchcodec).

Based on the tie_shoes_50_episodes_lora_no_adjust_1101 config.

Usage:
    python scripts/benchmark_video_backend.py --repo-id Vertax/xense_bi_arx5_tie_white_shoelaces_1030_no_adjust
    python scripts/benchmark_video_backend.py --repo-id Vertax/xense_bi_arx5_pick_and_place_cube --num-samples 500
"""

import argparse
from dataclasses import dataclass
import logging
import statistics
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result for a single backend."""

    backend: str
    total_time: float
    num_samples: int
    avg_time_per_sample: float
    samples_per_second: float
    min_time: float
    max_time: float
    std_time: float


def create_dataset(repo_id: str, video_backend: str | None, action_horizon: int = 30) -> LeRobotDataset:
    """Create a LeRobotDataset with the specified video backend."""
    logger.info(f"Creating dataset with video_backend={video_backend}")

    # Get metadata first
    dataset_meta = LeRobotDatasetMetadata(repo_id)

    # Calculate delta_timestamps
    delta_timestamps = {"action": [t / dataset_meta.fps for t in range(action_horizon)]}

    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )

    return dataset


def benchmark_dataset(
    dataset: LeRobotDataset,
    backend_name: str,
    num_samples: int = 100,
    warmup_samples: int = 10,
) -> BenchmarkResult:
    """Benchmark the dataset loading performance."""
    logger.info(f"Benchmarking {backend_name} backend with {num_samples} samples (warmup: {warmup_samples})")

    total_samples = len(dataset)
    if num_samples > total_samples:
        logger.warning(f"Requested {num_samples} samples but dataset only has {total_samples}. Using all samples.")
        num_samples = total_samples

    # Generate random indices for sampling
    indices = torch.randperm(total_samples)[: num_samples + warmup_samples].tolist()

    # Warmup phase
    logger.info(f"Warming up with {warmup_samples} samples...")
    for i in range(warmup_samples):
        _ = dataset[indices[i]]

    # Benchmark phase
    logger.info(f"Running benchmark with {num_samples} samples...")
    sample_times = []

    start_total = time.perf_counter()
    for i in range(warmup_samples, warmup_samples + num_samples):
        start_sample = time.perf_counter()
        _ = dataset[indices[i]]  # Load sample to measure decode time
        end_sample = time.perf_counter()
        sample_times.append(end_sample - start_sample)

        # Log progress every 100 samples
        if (i - warmup_samples + 1) % 100 == 0:
            logger.info(f"  Processed {i - warmup_samples + 1}/{num_samples} samples")

    end_total = time.perf_counter()
    total_time = end_total - start_total

    # Calculate statistics
    avg_time = statistics.mean(sample_times)
    std_time = statistics.stdev(sample_times) if len(sample_times) > 1 else 0.0
    min_time = min(sample_times)
    max_time = max(sample_times)
    samples_per_second = num_samples / total_time

    return BenchmarkResult(
        backend=backend_name,
        total_time=total_time,
        num_samples=num_samples,
        avg_time_per_sample=avg_time,
        samples_per_second=samples_per_second,
        min_time=min_time,
        max_time=max_time,
        std_time=std_time,
    )


def print_result(result: BenchmarkResult) -> None:
    """Print benchmark result."""
    print(f"\n{'=' * 60}")
    print(f"Backend: {result.backend}")
    print(f"{'=' * 60}")
    print(f"  Total time:           {result.total_time:.2f} seconds")
    print(f"  Number of samples:    {result.num_samples}")
    print(f"  Avg time per sample:  {result.avg_time_per_sample * 1000:.2f} ms")
    print(f"  Samples per second:   {result.samples_per_second:.2f}")
    print(f"  Min sample time:      {result.min_time * 1000:.2f} ms")
    print(f"  Max sample time:      {result.max_time * 1000:.2f} ms")
    print(f"  Std dev:              {result.std_time * 1000:.2f} ms")


def print_comparison(results: dict[str, BenchmarkResult]) -> None:
    """Print comparison between backends."""
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")

    # Create comparison table
    backends = list(results.keys())

    # Header
    print(f"\n{'Metric':<25}", end="")
    for backend in backends:
        print(f"{backend:>15}", end="")
    print()
    print("-" * (25 + 15 * len(backends)))

    # Rows
    metrics = [
        ("Total time (s)", lambda r: f"{r.total_time:.2f}"),
        ("Avg time/sample (ms)", lambda r: f"{r.avg_time_per_sample * 1000:.2f}"),
        ("Samples/second", lambda r: f"{r.samples_per_second:.2f}"),
        ("Min time (ms)", lambda r: f"{r.min_time * 1000:.2f}"),
        ("Max time (ms)", lambda r: f"{r.max_time * 1000:.2f}"),
        ("Std dev (ms)", lambda r: f"{r.std_time * 1000:.2f}"),
    ]

    for metric_name, metric_fn in metrics:
        print(f"{metric_name:<25}", end="")
        for backend in backends:
            print(f"{metric_fn(results[backend]):>15}", end="")
        print()

    # Winner determination
    if len(backends) == 2:
        r1, r2 = results[backends[0]], results[backends[1]]
        speedup = r1.samples_per_second / r2.samples_per_second if r2.samples_per_second > 0 else float("inf")

        if speedup > 1:
            winner = backends[0]
            speedup_ratio = speedup
        else:
            winner = backends[1]
            speedup_ratio = 1 / speedup if speedup > 0 else float("inf")

        print(f"\n{'=' * 60}")
        print(f"🏆 Winner: {winner}")
        print(f"   Speedup: {speedup_ratio:.2f}x faster than the other backend")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark video decoding backends for LeRobot datasets")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Vertax/xense_bi_arx5_tie_white_shoelaces_1030_no_adjust",
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of samples to benchmark (default: 200)",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=10,
        help="Number of warmup samples (default: 10)",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=30,
        help="Action horizon for delta_timestamps (default: 30)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["pyav", "torchcodec"],
        choices=["pyav", "torchcodec", "auto"],
        help="Backends to benchmark (default: pyav torchcodec)",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("LeRobot Video Backend Benchmark")
    print(f"{'=' * 60}")
    print(f"Dataset:        {args.repo_id}")
    print(f"Num samples:    {args.num_samples}")
    print(f"Warmup samples: {args.warmup_samples}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Backends:       {args.backends}")

    results = {}

    for backend in args.backends:
        # Map 'auto' to None (let LeRobot decide)
        video_backend = None if backend == "auto" else backend

        try:
            logger.info(f"\n{'=' * 40}")
            logger.info(f"Testing backend: {backend}")
            logger.info(f"{'=' * 40}")

            # Create dataset
            dataset = create_dataset(
                args.repo_id,
                video_backend=video_backend,
                action_horizon=args.action_horizon,
            )

            logger.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")

            # Run benchmark
            result = benchmark_dataset(
                dataset,
                backend_name=backend,
                num_samples=args.num_samples,
                warmup_samples=args.warmup_samples,
            )

            results[backend] = result
            print_result(result)

            # Clean up
            del dataset
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Failed to benchmark {backend}: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison if we have multiple results
    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
