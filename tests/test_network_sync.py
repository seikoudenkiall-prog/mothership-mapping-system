import numpy as np

from src.network_sync import Keyframe, NetworkSync


def test_keyframe_protocol_stays_below_50_kbps():
    frame = Keyframe(7, 0.7, (1.0, 2.0, 3.0), bytes(range(32)))
    packet = frame.encode()

    assert Keyframe.decode(packet) == frame
    assert NetworkSync.bandwidth_bps(keyframes_per_second=10) < 50_000
    assert len(packet) < 128


def test_disconnection_buffers_frames_and_reconnect_retries_packet_loss():
    sync = NetworkSync(packet_loss_rate=0.3, seed=17)
    sync.set_connected(False)
    frames = [
        Keyframe(index, index / 10, (index * 0.01, 0.0, 1.0), b"features")
        for index in range(100)
    ]

    for frame in frames:
        sync.send(frame)

    assert sync.pending_count == len(frames)
    sync.set_connected(True)
    for _ in range(20):
        sync.flush()

    assert sync.pending_count == 0
    assert [frame.sequence for frame in sync.received_keyframes] == list(range(100))


def test_reconnection_smoothing_keeps_each_axis_within_30mm():
    rng = np.random.default_rng(23)
    sync = NetworkSync(packet_loss_rate=0.3, seed=11)
    true_poses = {}
    drift = np.zeros(3)

    for sequence in range(101):
        true_pose = np.array([
            0.02 * sequence,
            np.sin(sequence / 12) * 0.4,
            1.0 + np.cos(sequence / 17) * 0.1,
        ])
        drift += rng.normal(0.0, 0.0007, size=3)
        estimated_pose = true_pose + drift
        true_poses[sequence] = true_pose
        if sequence == 35:
            sync.set_connected(False)
        sync.send(Keyframe(
            sequence, sequence / 10, tuple(estimated_pose), bytes([sequence % 256]) * 24
        ))
        if sequence == 75:
            sync.set_connected(True)
            for _ in range(20):
                sync.flush()

    for _ in range(20):
        sync.flush()
    assert sync.pending_count == 0

    for sequence in range(0, 101, 10):
        sync.add_anchor(sequence, true_poses[sequence] + rng.normal(0.0, 0.002, size=3))

    trajectory = sync.smoothed_trajectory()
    errors = np.array([
        trajectory[sequence] - true_pose
        for sequence, true_pose in true_poses.items()
    ])
    max_axis_errors_mm = np.max(np.abs(errors), axis=0) * 1000

    assert np.all(max_axis_errors_mm < 30.0), max_axis_errors_mm