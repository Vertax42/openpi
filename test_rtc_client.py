import time
import numpy as np
import logging
from openpi_client import websocket_client_policy
from openpi_client import rtc_action_chunk_broker

# Configure logging to see broker messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_rtc_client():
    print("Connecting to policy server at ws://0.0.0.0:8000 ...")
    try:
        policy = websocket_client_policy.WebsocketClientPolicy(
            host="0.0.0.0", port=8000
        )
        print(f"Connected! Server Metadata: {policy.get_server_metadata()}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Initializing RTC Broker...")
    # We use a lower frequency for testing to allow inference to keep up if running on CPU
    # In real deployment with GPU, this should be 10Hz-50Hz
    TEST_HZ = 30.0
    broker = rtc_action_chunk_broker.RTCActionChunkBroker(
        policy=policy,
        frequency_hz=TEST_HZ,
        action_queue_size_to_get_new_actions=30,
        rtc_enabled=True,
        execution_horizon=20,  # Adjusted to 15 (smaller than action_horizon=30)
    )

    # Create fake observation
    obs = {
        "images": {
            "cam_high": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8),
        },
        "state": np.random.randn(14).astype(np.float32),
        "prompt": "tie shoelaces",
    }

    print(f"Starting control loop simulation ({TEST_HZ} Hz) for 50 steps...")

    try:
        # Warmup / Fill Queue
        # The broker starts the thread on the first infer call
        print("Sending initial observation to start background thread...")

        for i in range(1000):
            loop_start = time.time()

            # Update obs slightly
            obs["state"] += np.random.randn(14).astype(np.float32) * 0.01

            try:
                # infer() updates the latest observation for the background thread
                # and retrieves the next action from the queue
                result = broker.infer(obs)
                action = result["actions"]

                # Check action validity
                if action is not None:
                    print(
                        f"Step {i:02d}: Action executed. Shape: {action.shape} | First dim: {action[0]:.4f}"
                    )
                else:
                    print(f"Step {i:02d}: Warning: Got None action")

            except RuntimeError as e:
                print(f"Step {i:02d}: Error: {e}")
                print(
                    "Queue might be empty because inference is slower than control loop."
                )

            # Maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / TEST_HZ) - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        print("Stopping broker...")
        broker.stop()
        print("Test finished.")


if __name__ == "__main__":
    test_rtc_client()
