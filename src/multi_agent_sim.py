from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class _Agent:
    position: np.ndarray
    estimate: np.ndarray
    tasks: list
    task_index: int = 0
    mode: str = "scan"
    covariance: float = 0.001
    hop_count: int = 0
    distance_travelled: float = 0.0
    samples: dict = field(default_factory=dict)


def _safe_step(position, goal, other_positions, safety_distance, step_size):
    direction = goal - position
    distance = np.linalg.norm(direction)
    if distance < 1e-9:
        return position.copy()

    preferred = direction / distance
    directions = [preferred]
    for angle in (45, -45, 90, -90, 135, -135, 180):
        radians = np.deg2rad(angle)
        directions.append(np.array([
            preferred[0] * np.cos(radians) - preferred[1] * np.sin(radians),
            preferred[0] * np.sin(radians) + preferred[1] * np.cos(radians),
            preferred[2],
        ]))

    candidates = []
    for candidate_direction in directions:
        norm = np.linalg.norm(candidate_direction)
        if norm < 1e-9:
            continue
        candidate = position + candidate_direction / norm * min(step_size, distance)
        if all(np.linalg.norm(candidate - other) >= safety_distance - 1e-9
               for other in other_positions):
            candidates.append(candidate)

    if not candidates:
        return position.copy()
    return min(candidates, key=lambda candidate: np.linalg.norm(goal - candidate))


def run_multi_agent_simulation(agent_count=2, safety_distance=1.5, seed=42,
                               max_steps=5000, step_duration_s=0.1):
    """Simulate cooperative scanning; one agent is supported as a timing baseline."""
    if agent_count < 1:
        raise ValueError("agent_count must be at least 1")
    if safety_distance <= 0:
        raise ValueError("safety_distance must be positive")
    if not np.isfinite(step_duration_s) or step_duration_s <= 0:
        raise ValueError("step_duration_s must be positive and finite")

    rng = np.random.default_rng(seed)
    obstacles = [
        (np.array([15.0, 15.0]), 1.8),
        (np.array([30.0, 25.0]), 2.0),
        (np.array([20.0, 32.0]), 1.4),
    ]
    tasks_by_obstacle = []
    for obstacle_id, (center, height) in enumerate(obstacles):
        tasks_by_obstacle.append([
            (obstacle_id * 2, np.array([center[0], center[1], height]), center),
            (obstacle_id * 2 + 1,
             np.array([center[0] + 0.5, center[1], height / 2]), center),
        ])

    assignments = [[] for _ in range(agent_count)]
    for obstacle_id, tasks in enumerate(tasks_by_obstacle):
        assignments[obstacle_id % agent_count].extend(tasks)

    agents = []
    for agent_id, tasks in enumerate(assignments):
        start = np.array([
            22.5 + (agent_id - (agent_count - 1) / 2) * 2.0,
            22.0,
            1.0,
        ])
        agents.append(_Agent(start, start.copy(), tasks))

    elapsed_steps = 0
    min_separation = min(
        (np.linalg.norm(agents[i].position - agents[j].position)
         for i in range(agent_count) for j in range(i + 1, agent_count)),
        default=float("inf"),
    )

    while any(agent.mode != "done" for agent in agents):
        if elapsed_steps >= max_steps:
            raise RuntimeError("simulation did not finish before max_steps")

        goals = []
        for agent in agents:
            if agent.mode == "hop":
                goals.append(np.array([agent.position[0], agent.position[1], 2.2]))
            elif agent.task_index < len(agent.tasks):
                _, point, _ = agent.tasks[agent.task_index]
                goals.append(point + np.array([0.0, 0.9, 0.0]))
            else:
                agent.mode = "done"
                goals.append(agent.position.copy())

        for offset in range(agent_count):
            agent_id = (elapsed_steps + offset) % agent_count
            agent = agents[agent_id]
            if agent.mode == "done":
                continue
            other_positions = [other.position for i, other in enumerate(agents)
                               if i != agent_id]
            previous_position = agent.position.copy()
            agent.position = _safe_step(
                agent.position, goals[agent_id], other_positions,
                safety_distance, 0.3,
            )
            displacement = agent.position - previous_position
            agent.distance_travelled += float(np.linalg.norm(displacement))
            agent.estimate += displacement + rng.normal(0.0, 0.0015, size=3)

            if agent.mode == "hop" and agent.position[2] >= 2.15:
                marker_position = agent.position + rng.normal(0.0, 0.003, size=3)
                agent.estimate += 0.95 * (marker_position - agent.estimate)
                agent.covariance = 0.001
                agent.hop_count += 1
                agent.mode = "scan"
            elif agent.mode == "scan":
                task_id, point, center = agent.tasks[agent.task_index]
                is_nlos = (agent.position[2] <= 1.2 and
                           np.linalg.norm(agent.position[:2] - center) < 2.0)
                if not is_nlos:
                    marker_position = agent.position + rng.normal(0.0, 0.003, size=3)
                    agent.estimate += 0.95 * (marker_position - agent.estimate)

                scan_position = point + np.array([0.0, 0.9, 0.0])
                if np.linalg.norm(agent.position - scan_position) <= 0.05:
                    ray = point - agent.position
                    measurement = agent.estimate + ray + rng.normal(0.0, 0.002, size=3)
                    agent.samples.setdefault(task_id, []).append(measurement)
                    if is_nlos:
                        agent.covariance += 0.04
                    if len(agent.samples[task_id]) >= 3:
                        agent.task_index += 1
                    if agent.covariance >= 0.12:
                        agent.mode = "hop"

        for i in range(agent_count):
            for j in range(i + 1, agent_count):
                distance = np.linalg.norm(agents[i].position - agents[j].position)
                min_separation = min(min_separation, distance)
        elapsed_steps += 1

    errors = []
    for agent in agents:
        for task_id, point, _ in agent.tasks:
            samples = agent.samples.get(task_id, [])
            if len(samples) < 3:
                continue
            difference = np.median(samples, axis=0) - point
            errors.append({
                "point": task_id,
                "xy": np.linalg.norm(difference[:2]) * 1000.0,
                "z": abs(difference[2]) * 1000.0,
                "3d": np.linalg.norm(difference) * 1000.0,
            })

    return {
        "errors": pd.DataFrame(errors, columns=["point", "xy", "z", "3d"]),
        "elapsed_steps": elapsed_steps,
        "elapsed_time_s": elapsed_steps * step_duration_s,
        "step_duration_s": step_duration_s,
        "distance_by_agent_m": [agent.distance_travelled for agent in agents],
        "total_distance_m": sum(agent.distance_travelled for agent in agents),
        "min_separation": min_separation,
        "hop_counts": [agent.hop_count for agent in agents],
        "scan_complete": len(errors) == sum(map(len, assignments)),
    }