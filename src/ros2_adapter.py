from dataclasses import dataclass
import struct

import numpy as np

from src.network_sync import Keyframe


_FLOAT_FIELD_FORMATS = {7: "f4", 8: "f8"}


@dataclass(frozen=True)
class PoseStampedData:
    timestamp: float
    frame_id: str
    position: tuple
    orientation: tuple


@dataclass(frozen=True)
class ZenohPacket:
    topic: str
    payload: bytes


class Ros2Adapter:
    """Convert ROS 2 pose and point-cloud messages to compact keyframes."""

    def __init__(self, topic_prefix="mothership/keyframes"):
        self.topic_prefix = topic_prefix.strip("/")
        if not self.topic_prefix:
            raise ValueError("topic_prefix must not be empty")

    @staticmethod
    def pose_stamped_to_data(message):
        header = message.header
        stamp = header.stamp
        seconds = int(getattr(stamp, "sec", getattr(stamp, "secs", 0)))
        nanoseconds = int(getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0)))
        if not 0 <= nanoseconds < 1_000_000_000:
            raise ValueError("stamp nanoseconds must be in [0, 1e9)")

        ros_pose = message.pose
        position = np.array([
            ros_pose.position.x, ros_pose.position.y, ros_pose.position.z,
        ], dtype=float)
        orientation = np.array([
            ros_pose.orientation.x, ros_pose.orientation.y,
            ros_pose.orientation.z, ros_pose.orientation.w,
        ], dtype=float)
        orientation_norm = np.linalg.norm(orientation)
        if not np.isfinite(position).all() or not np.isfinite(orientation).all():
            raise ValueError("pose coordinates must be finite")
        if orientation_norm < 1e-12:
            raise ValueError("pose orientation must have non-zero length")

        return PoseStampedData(
            timestamp=seconds + nanoseconds * 1e-9,
            frame_id=str(header.frame_id),
            position=tuple(position),
            orientation=tuple(orientation / orientation_norm),
        )

    @staticmethod
    def pointcloud2_to_points(message):
        if hasattr(message, "points"):
            points = np.asarray(message.points, dtype=float)
            if points.ndim != 2 or points.shape[1] < 3:
                raise ValueError("points must have at least three columns")
            points = points[:, :3]
        else:
            fields = {field.name: field for field in message.fields}
            if any(name not in fields for name in ("x", "y", "z")):
                raise ValueError("PointCloud2 must contain x, y, and z fields")

            point_step = int(message.point_step)
            width = int(message.width)
            height = int(message.height)
            row_step = int(getattr(message, "row_step", width * point_step))
            if point_step <= 0 or width <= 0 or height <= 0:
                raise ValueError("PointCloud2 dimensions and point_step must be positive")

            endian = ">" if getattr(message, "is_bigendian", False) else "<"
            formats = []
            offsets = []
            for name in ("x", "y", "z"):
                field = fields[name]
                format_code = _FLOAT_FIELD_FORMATS.get(int(field.datatype))
                if format_code is None:
                    raise ValueError(f"PointCloud2 field {name} must be float32 or float64")
                formats.append(endian + format_code)
                offsets.append(int(field.offset))
            if any(offset < 0 or offset + np.dtype(fmt).itemsize > point_step
                   for offset, fmt in zip(offsets, formats)):
                raise ValueError("PointCloud2 field exceeds point_step")

            try:
                data = memoryview(message.data)
            except TypeError:
                data = memoryview(bytes(message.data))
            required_bytes = (height - 1) * row_step + width * point_step
            if len(data) < required_bytes:
                raise ValueError("PointCloud2 data is shorter than its dimensions")

            dtype = np.dtype({
                "names": ["x", "y", "z"],
                "formats": formats,
                "offsets": offsets,
                "itemsize": point_step,
            })
            cloud = np.ndarray(
                (height, width), dtype=dtype, buffer=data,
                strides=(row_step, point_step),
            )
            points = np.column_stack((
                cloud["x"].reshape(-1),
                cloud["y"].reshape(-1),
                cloud["z"].reshape(-1),
            ))

        points = np.asarray(points, dtype=float)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError("point cloud contains no finite points")
        return points

    @staticmethod
    def geometric_features(points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("points must be a non-empty Nx3 array")
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError("point cloud contains no finite points")

        centroid = points.mean(axis=0)
        spread = points.std(axis=0)
        point_count = float(len(points))
        extent = float(np.linalg.norm(np.ptp(points, axis=0)))
        return struct.pack("!8f", *centroid, *spread, point_count, extent)

    def to_keyframe(self, pose_message, pointcloud_message, sequence):
        pose = self.pose_stamped_to_data(pose_message)
        points = self.pointcloud2_to_points(pointcloud_message)
        descriptor = self.geometric_features(points)
        orientation = struct.pack("!4f", *pose.orientation)
        return Keyframe(
            sequence=sequence,
            timestamp=pose.timestamp,
            pose=pose.position,
            features=orientation + descriptor,
        )

    def to_zenoh_packet(self, pose_message, pointcloud_message, sequence):
        pose = self.pose_stamped_to_data(pose_message)
        keyframe = self.to_keyframe(pose_message, pointcloud_message, sequence)
        topic = f"{self.topic_prefix}/{pose.frame_id}/{sequence}"
        return ZenohPacket(topic=topic, payload=keyframe.encode())

    def publish_zenoh(self, session, pose_message, pointcloud_message, sequence):
        packet = self.to_zenoh_packet(pose_message, pointcloud_message, sequence)
        session.put(packet.topic, packet.payload)
        return packet