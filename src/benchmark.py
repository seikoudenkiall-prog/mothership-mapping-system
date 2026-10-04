import json
import math
from pathlib import Path

from src.multi_agent_sim import run_multi_agent_simulation
from src.network_sync import Keyframe


def run_benchmark(agent_count=2, seed=42, step_duration_s=0.1,
                  keyframe_interval_steps=10):
    if keyframe_interval_steps < 1:
        raise ValueError("keyframe_interval_steps must be at least 1")

    simulation = run_multi_agent_simulation(
        agent_count=agent_count,
        seed=seed,
        step_duration_s=step_duration_s,
    )
    if not simulation["scan_complete"] or simulation["errors"].empty:
        raise RuntimeError("benchmark simulation did not complete all scan tasks")

    sample_keyframe = Keyframe(0, 0.0, (0.0, 0.0, 0.0), bytes(48))
    payload_bytes = len(sample_keyframe.encode())
    frames_per_agent = math.ceil(
        simulation["elapsed_steps"] / keyframe_interval_steps
    )
    keyframes_sent = frames_per_agent * agent_count
    total_bytes = keyframes_sent * payload_bytes
    duration = simulation["elapsed_time_s"]
    errors = simulation["errors"]

    return {
        "environment": {
            "area_m2": 2000,
            "max_obstacle_height_m": 2.0,
        },
        "simulation": {
            "agent_count": agent_count,
            "elapsed_steps": simulation["elapsed_steps"],
            "step_duration_s": step_duration_s,
            "elapsed_time_s": duration,
            "total_distance_m": simulation["total_distance_m"],
            "distance_by_agent_m": simulation["distance_by_agent_m"],
        },
        "communication": {
            "keyframe_interval_steps": keyframe_interval_steps,
            "keyframes_sent": keyframes_sent,
            "payload_bytes_per_keyframe": payload_bytes,
            "total_bytes": total_bytes,
            "throughput_kbps": total_bytes * 8 / duration / 1000,
        },
        "accuracy_mm": {
            "max_xy": float(errors["xy"].max()),
            "max_z": float(errors["z"].max()),
            "max_3d": float(errors["3d"].max()),
        },
    }


def format_markdown_report(report):
    simulation = report["simulation"]
    communication = report["communication"]
    accuracy = report["accuracy_mm"]
    environment = report["environment"]
    agent_distances = ", ".join(
        f"{distance:.2f}" for distance in simulation["distance_by_agent_m"]
    )
    return "\n".join([
        "# Mapping System Benchmark",
        "",
        f"- Environment: {environment['area_m2']} m2, "
        f"max obstacle height {environment['max_obstacle_height_m']:.1f} m",
        f"- Agents: {simulation['agent_count']}",
        f"- Elapsed time: {simulation['elapsed_time_s']:.2f} s "
        f"({simulation['elapsed_steps']} steps)",
        f"- Total travel distance: {simulation['total_distance_m']:.2f} m",
        f"- Distance by agent: {agent_distances} m",
        f"- Communication: {communication['total_bytes']} bytes, "
        f"{communication['throughput_kbps']:.2f} kbps",
        "",
        "| Accuracy metric | Maximum error |",
        "| --- | ---: |",
        f"| XY | {accuracy['max_xy']:.2f} mm |",
        f"| Z | {accuracy['max_z']:.2f} mm |",
        f"| 3D | {accuracy['max_3d']:.2f} mm |",
        "",
    ])


def generate_benchmark_report(output_dir=".", agent_count=2, seed=42,
                              step_duration_s=0.1,
                              keyframe_interval_steps=10):
    report = run_benchmark(
        agent_count=agent_count,
        seed=seed,
        step_duration_s=step_duration_s,
        keyframe_interval_steps=keyframe_interval_steps,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "benchmark_report.md").write_text(
        format_markdown_report(report), encoding="utf-8",
    )
    return report