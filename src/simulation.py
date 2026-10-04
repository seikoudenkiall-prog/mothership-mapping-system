import numpy as np
import pandas as pd

def run_simulation(steps=300):
    np.random.seed(42)
    # 親機: Z=3.5m, 障害物: 2.0m以内, 敷地: 2000平米
    mothership_pos = np.array([22.5, 22.0, 3.5])
    drone_true = np.array([22.5, 22.0, 1.0])
    drone_est = drone_true.copy()
    accum_cov = 0.001

    obstacles = [
        {'id': 1, 'center': np.array([15.0, 15.0]), 'height': 1.8},
        {'id': 2, 'center': np.array([30.0, 25.0]), 'height': 2.0},
        {'id': 3, 'center': np.array([20.0, 32.0]), 'height': 1.4}
    ]

    eval_points = []
    for obs in obstacles:
        c = obs['center']
        h = obs['height']
        eval_points.append({'pt': np.array([c[0], c[1], h]), 'samples': []})
        eval_points.append({'pt': np.array([c[0] + 0.5, c[1], h/2]), 'samples': []})

    wp_idx = 0
    in_escape = False

    for t in range(steps):
        target = obstacles[wp_idx % len(obstacles)]
        c = target['center']
        is_nlos = (drone_true[2] <= 1.2 and np.linalg.norm(drone_true[:2] - c) < 2.0)

        # 30mm制限のための共分散リミット
        if accum_cov >= 0.12:
            in_escape = True

        if in_escape:
            goal = np.array([drone_true[0], drone_true[1], 2.2])
            if drone_true[2] >= 2.15:
                in_escape = False
                wp_idx += 1
        else:
            goal = np.array([c[0] + 0.6, c[1], 0.8])
            if np.linalg.norm(drone_true[:2] - c) < 1.0:
                if t % 25 == 0:
                    wp_idx += 1

        direction = goal - drone_true
        d = np.linalg.norm(direction)
        vel = (direction / d) * min(0.3, d) if d > 1e-3 else np.zeros(3)
        drone_true += vel

        # 局所オドメトリノイズ
        drone_est += vel + np.random.normal(0, 0.0015, size=3)

        if is_nlos:
            accum_cov += 0.012 * np.linalg.norm(vel)
        else:
            accum_cov = max(0.001, accum_cov * 0.05)
            # 親機の4K光学トラッキング補正 (3mmノイズ)
            marker_noise = np.random.normal(0, 0.003, size=3)
            drone_est += 0.95 * ((drone_true + marker_noise) - drone_est)

        # センサによる観測 (TSDF用サンプル蓄積)
        for ep in eval_points:
            dist = np.linalg.norm(drone_true - ep['pt'])
            if dist <= 1.2:
                ray = ep['pt'] - drone_true
                ep['samples'].append(drone_est + ray + np.random.normal(0, 0.002, size=3))

    errors = []
    for ep in eval_points:
        if len(ep['samples']) >= 3:
            fused = np.median(ep['samples'], axis=0)
            diff = fused - ep['pt']
            err_xy = np.linalg.norm(diff[:2]) * 1000.0
            err_z = abs(diff[2]) * 1000.0
            err_3d = np.linalg.norm(diff) * 1000.0
            errors.append({'xy': err_xy, 'z': err_z, '3d': err_3d})

    return pd.DataFrame(errors)

if __name__ == '__main__':
    df = run_simulation()
    print("--- Simulation Result ---")
    print(f"Max 3D Error: {df['3d'].max():.2f} mm")
    print(f"Max XY Error: {df['xy'].max():.2f} mm")
    print(f"Max Z Error:  {df['z'].max():.2f} mm")
