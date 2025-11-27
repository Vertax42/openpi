import numpy as np
import time
from openpi_client import websocket_client_policy


def test_rtc():
    print("Connecting to policy server at ws://0.0.0.0:8000 ...")
    try:
        # Connect to server
        policy = websocket_client_policy.WebsocketClientPolicy(
            host="0.0.0.0", port=8000
        )
        print(f"Connected! Server Metadata: {policy.get_server_metadata()}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print(
            "Make sure the policy server is running (python scripts/serve_policy.py ...)"
        )
        return

    # Create fake observation matches pi05_base_arx5_tie_shoes_lora config
    obs = {
        "images": {
            "cam_high": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        },
        "state": np.random.randn(14).astype(np.float32),
        "prompt": "tie shoelaces",
    }

    print("\n--- 1. Standard Inference (No RTC) ---")
    start = time.time()
    result1 = policy.infer(obs)
    print(f"Latency: {(time.time() - start)*1000:.2f} ms")
    actions1 = result1["actions"]
    print(f"Actions shape: {actions1.shape}")

    print("\n--- 2. Continuous RTC Inference (50 steps) ---")
    latencies = []

    # We'll use the output from standard inference as a base for prev_chunk
    base_actions = actions1.copy()

    for i in range(50):
        # Simulate observation change
        obs["state"] += np.random.randn(14).astype(np.float32) * 0.01

        # Simulate varying delay (1, 2, 3, 1, 2, 3...) to check if it triggers recompilation
        inference_delay = (i % 3) + 1

        # Simulate prev_chunk_left_over
        prev_chunk_left_over = base_actions + np.random.randn(*base_actions.shape) * 0.1

        rtc_kwargs = {
            "prev_chunk_left_over": prev_chunk_left_over,
            "inference_delay": inference_delay,
            "execution_horizon": 10,
        }

        start = time.time()
        result = policy.infer(obs, **rtc_kwargs)
        end = time.time()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # Simple check to see if result is valid
        actions = result["actions"]
        actions_original = result["actions_original"]
        # print(f"Actions data: {actions}")
        # print(f"Actions original data: {actions_original}")

        print(
            f"Step {i+1:02d}/50: Latency = {latency_ms:8.2f} ms | Delay param: {inference_delay}"
        )

    # Statistics
    warmup = 3
    if len(latencies) > warmup:
        valid_latencies = latencies[warmup:]
        avg_latency = np.mean(valid_latencies)
        p50 = np.percentile(valid_latencies, 50)
        p99 = np.percentile(valid_latencies, 99)
        print(f"\nStatistics (excluding first {warmup} steps):")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  Median (P50): {p50:.2f} ms")
        print(f"  P99: {p99:.2f} ms")

        if avg_latency > 1000:
            print("\n⚠️  WARNING: High latency detected.")
            print("Possible causes:")
            print("1. Running on CPU instead of GPU.")
            print(
                "2. JAX is recompiling every step (check if inference_delay is being treated as static)."
            )
    else:
        print("Not enough steps for statistics.")


if __name__ == "__main__":
    test_rtc()
