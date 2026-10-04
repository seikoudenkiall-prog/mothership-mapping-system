import pytest
from src.simulation import run_simulation

def test_mapping_accuracy_under_30mm():
    df = run_simulation()
    
    max_xy = df['xy'].max()
    max_z = df['z'].max()
    max_3d = df['3d'].max()
    
    print(f"\n[Test Result] Max XY: {max_xy:.2f}mm, Max Z: {max_z:.2f}mm, Max 3D: {max_3d:.2f}mm")
    
    # 全軸30mm以下の検証アサーション
    assert max_xy < 30.0, f"XY Error exceeded 30mm: {max_xy}mm"
    assert max_z < 30.0, f"Z Error exceeded 30mm: {max_z}mm"
    assert max_3d < 30.0, f"3D Euclidean Error exceeded 30mm: {max_3d}mm"
