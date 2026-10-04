from collections import OrderedDict
from dataclasses import dataclass
import math
import struct

import gtsam
import numpy as np


_HEADER = struct.Struct("!Id3fH")


@dataclass(frozen=True)
class Keyframe:
    sequence: int
    timestamp: float
    pose: tuple
    features: bytes

    def __post_init__(self):
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if len(self.pose) != 3 or not np.isfinite(self.pose).all():
            raise ValueError("pose must contain three finite coordinates")
        if len(self.features) > 65535:
            raise ValueError("feature descriptor is too large")

    def encode(self):
        return _HEADER.pack(
            self.sequence, self.timestamp, *self.pose, len(self.features)
        ) + self.features

    @classmethod
    def decode(cls, packet):
        if len(packet) < _HEADER.size:
            raise ValueError("packet is shorter than the keyframe header")
        sequence, timestamp, x, y, z, feature_length = _HEADER.unpack_from(packet)
        if len(packet) != _HEADER.size + feature_length:
            raise ValueError("packet length does not match feature descriptor")
        return cls(sequence, timestamp, (x, y, z), packet[_HEADER.size:])


class _TrajectorySmoother:
    def __init__(self):
        self._poses = {}
        self._anchors = {}
        self._isam = gtsam.ISAM2()

    def add_keyframe(self, keyframe):
        if keyframe.sequence in self._poses:
            return
        out_of_order = bool(self._poses) and keyframe.sequence < max(self._poses)
        self._poses[keyframe.sequence] = np.asarray(keyframe.pose, dtype=float)
        if out_of_order:
            self._rebuild()
            return

        sequence = keyframe.sequence
        pose = self._pose(self._poses[sequence])
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        key = self._key(sequence)
        initial.insert(key, pose)
        previous = [item for item in self._poses if item < sequence]
        if previous:
            previous_sequence = max(previous)
            previous_pose = self._pose(self._poses[previous_sequence])
            relative_pose = previous_pose.between(pose)
            graph.add(gtsam.BetweenFactorPose3(
                self._key(previous_sequence), key, relative_pose,
                self._odometry_noise(),
            ))
        else:
            graph.add(gtsam.PriorFactorPose3(
                key, pose, self._pose_noise(1.0, 1e-3)
            ))
        self._isam.update(graph, initial)

    def add_anchor(self, sequence, pose):
        anchor = np.asarray(pose, dtype=float)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("anchor pose must contain three finite coordinates")
        self._anchors[sequence] = anchor
        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.PriorFactorPose3(
            self._key(sequence), self._pose(anchor), self._pose_noise(0.002, 1e-3)
        ))
        self._isam.update(graph, gtsam.Values())

    def trajectory(self):
        if not self._poses:
            return {}
        estimate = self._isam.calculateEstimate()
        return {
            sequence: np.asarray(estimate.atPose3(self._key(sequence)).translation())
            for sequence in sorted(self._poses)
        }

    def _rebuild(self):
        self._isam = gtsam.ISAM2()
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        sequences = sorted(self._poses)
        for index, sequence in enumerate(sequences):
            pose = self._pose(self._poses[sequence])
            key = self._key(sequence)
            initial.insert(key, pose)
            if index == 0:
                graph.add(gtsam.PriorFactorPose3(
                    key, pose, self._pose_noise(1.0, 1e-3)
                ))
            else:
                previous = sequences[index - 1]
                previous_pose = self._pose(self._poses[previous])
                graph.add(gtsam.BetweenFactorPose3(
                    self._key(previous), key, previous_pose.between(pose),
                    self._odometry_noise(),
                ))
        for sequence, anchor in self._anchors.items():
            if sequence in self._poses:
                graph.add(gtsam.PriorFactorPose3(
                    self._key(sequence), self._pose(anchor),
                    self._pose_noise(0.002, 1e-3),
                ))
        self._isam.update(graph, initial)

    @staticmethod
    def _key(sequence):
        return gtsam.symbol("x", sequence)

    @staticmethod
    def _pose(position):
        return gtsam.Pose3(gtsam.Rot3(), np.asarray(position, dtype=float))

    @staticmethod
    def _pose_noise(translation_sigma, rotation_sigma):
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([
            rotation_sigma, rotation_sigma, rotation_sigma,
            translation_sigma, translation_sigma, translation_sigma,
        ]))

    @classmethod
    def _odometry_noise(cls):
        return cls._pose_noise(0.003, 1e-3)


class NetworkSync:
    """Reliable keyframe transport with an offline buffer and pose smoothing."""

    def __init__(self, packet_loss_rate=0.3, seed=42, connected=True):
        if not 0.0 <= packet_loss_rate <= 1.0:
            raise ValueError("packet_loss_rate must be between 0 and 1")
        self.packet_loss_rate = packet_loss_rate
        self.connected = connected
        self._rng = np.random.default_rng(seed)
        self._pending = OrderedDict()
        self._received = {}
        self._smoother = _TrajectorySmoother()

    @property
    def pending_count(self):
        return len(self._pending)

    @property
    def received_keyframes(self):
        return [self._received[key] for key in sorted(self._received)]

    def set_connected(self, connected):
        self.connected = bool(connected)
        return self.flush() if self.connected else 0

    def send(self, keyframe):
        packet = keyframe.encode()
        existing = self._pending.get(keyframe.sequence)
        if existing is not None and existing != packet:
            raise ValueError("sequence number already contains different data")
        self._pending.setdefault(keyframe.sequence, packet)
        return self.flush() if self.connected else 0

    def flush(self, max_attempts=1):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not self.connected:
            return 0

        delivered = 0
        for sequence, packet in list(self._pending.items()):
            for _ in range(max_attempts):
                if self._rng.random() < self.packet_loss_rate:
                    continue
                keyframe = Keyframe.decode(packet)
                self._received.setdefault(sequence, keyframe)
                self._smoother.add_keyframe(keyframe)
                del self._pending[sequence]
                delivered += 1
                break
        return delivered

    def add_anchor(self, sequence, pose):
        if sequence not in self._received:
            raise KeyError("anchor must refer to a received keyframe")
        self._smoother.add_anchor(sequence, pose)

    def smoothed_trajectory(self):
        return self._smoother.trajectory()

    @staticmethod
    def bandwidth_bps(keyframes_per_second, feature_bytes=32):
        if keyframes_per_second < 0 or feature_bytes < 0:
            raise ValueError("rate and feature size must be non-negative")
        return (_HEADER.size + feature_bytes) * keyframes_per_second * 8