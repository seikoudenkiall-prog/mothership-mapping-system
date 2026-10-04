from src.multi_agent_sim import run_multi_agent_simulation


def test_two_agents_preserve_accuracy_and_reduce_scan_time():
    single_agent = run_multi_agent_simulation(agent_count=1)
    two_agents = run_multi_agent_simulation(agent_count=2)

    assert two_agents["scan_complete"]
    assert two_agents["errors"]["xy"].max() < 30.0
    assert two_agents["errors"]["z"].max() < 30.0
    assert two_agents["errors"]["3d"].max() < 30.0
    assert two_agents["elapsed_steps"] < single_agent["elapsed_steps"] * 0.75


def test_agents_avoid_collisions_and_resynchronize_independently():
    result = run_multi_agent_simulation(agent_count=2)

    assert result["min_separation"] >= 1.5
    assert all(hop_count > 0 for hop_count in result["hop_counts"])