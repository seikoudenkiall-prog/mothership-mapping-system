from types import SimpleNamespace
import struct
import time

import numpy as np

from src.network_sync import Keyframe
from src.ros2_adapter import Ros2Adapter


def _pose_stamped(position=(1.012, 2.009, 3.025)):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nanosec=345_000_000), frame_id="map"
        ),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def _pointcloud2(points, field_names=("x", "y", "z", "intensity")):
    formats = {"x": "f", "y": "f", "z": "f", "intensity": "f", "rgb": "I"}
    fields = [
        SimpleNamespace(name=name, offset=index * 4, datatype=7 if name != "rgb" else 6)
        for index, name in enumerate(field_names)
    ]
    packed = b"".join(
        struct.pack("<" + "".join(formats[name] for name in field_names), *values)
        for values in points
    )
    return SimpleNamespace(
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=len(field_names) * 4,
        row_step=len(points) * len(field_names) * 4,
        data=packed,
    )


def test_livox_and_rgbd_pointclouds_become_compact_zenoh_keyframes():
    adapter = Ros2Adapter()
    points = [(0.1 * index, 0.2, 1.0, 42) for index in range(20)]
    pose = _pose_stamped()

    for cloud in (
        _pointcloud2(points),
        _pointcloud2([point[:3] + (0x112233,) for point in points],
                     field_names=("x", "y", "z", "rgb")),
    ):
        packet = adapter.to_zenoh_packet(pose, cloud, sequence=4)
        keyframe = Keyframe.decode(packet.payload)
        assert packet.topic == "mothership/keyframes/map/4"
        assert keyframe.timestamp == 12.345
        assert len(keyframe.features) == 48
        assert len(packet.payload) < 100
        assert np.allclose(keyframe.pose, (1.012, 2.009, 3.025), atol=1e-6)


def test_conversion_latency_and_pose_accuracy_stay_within_bounds():
    rng = np.random.default_rng(31)
    xyz = rng.normal(size=(4096, 3)).astype(np.float32)
    cloud = _pointcloud2([tuple(point) + (10.0,) for point in xyz])
    adapter = Ros2Adapter()

    started = time.perf_counter()
    packet = adapter.to_zenoh_packet(_pose_stamped(), cloud, sequence=1)
    elapsed = time.perf_counter() - started

    decoded = Keyframe.decode(packet.payload)
    axis_error_mm = np.abs(np.asarray(decoded.pose) - (1.0, 2.0, 3.0)) * 1000
    assert elapsed < 0.25
    assert np.all(axis_error_mm < 30.0)


def test_zenoh_publish_uses_compact_payload():
    class Session:
        def __init__(self):
            self.published = None

        def put(self, topic, payload):
            self.published = (topic, payload)

    session = Session()
    packet = Ros2Adapter().publish_zenoh(
        session, _pose_stamped(), _pointcloud2([(1.0, 2.0, 3.0, 4.0)]), 9
    )

    assert session.published == (packet.topic, packet.payload)
    assert len(Keyframe.decode(session.published[1]).features) == 48