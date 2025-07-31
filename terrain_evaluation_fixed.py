"""
地形汎化性能評価ツール（修正版）
段差と傾斜の3段階で既存ポリシーの性能を評価
"""

import argparse
import time
import numpy as np
import torch
import genesis as gs


def create_terrain(terrain_type: str, difficulty: int):
    """地形を作成
    Args:
        terrain_type: "steps" or "slopes" 
        difficulty: 1 (easy), 2 (medium), 3 (hard)
    """
    # 段差設定
    step_heights = {1: -0.05, 2: -0.10, 3: -0.15}  # 負の値で段差を作成
    step_widths = {1: 1.0, 2: 0.8, 3: 0.6}
    
    # 傾斜設定
    slopes = {1: -0.1, 2: -0.2, 3: -0.3}  # 負の値で傾斜を作成
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 3.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
        ),
        show_viewer=True,
    )
    
    if terrain_type == "steps":
        # カスタムパラメータを個別に設定
        terrain = scene.add_entity(
            morph=gs.morphs.Terrain(
                n_subterrains=(3, 3),
                subterrain_size=(4.0, 4.0), 
                horizontal_scale=0.25,
                vertical_scale=0.01,
                subterrain_types=[
                    ["flat_terrain", "pyramid_stairs_terrain", "flat_terrain"],
                    ["pyramid_stairs_terrain", "pyramid_stairs_terrain", "pyramid_stairs_terrain"],
                    ["flat_terrain", "pyramid_stairs_terrain", "flat_terrain"],
                ],
                pyramid_stairs_terrain={
                    "step_height": step_heights[difficulty],
                    "step_width": step_widths[difficulty]
                }
            ),
        )
    
    elif terrain_type == "slopes":
        terrain = scene.add_entity(
            morph=gs.morphs.Terrain(
                n_subterrains=(3, 3),
                subterrain_size=(4.0, 4.0),
                horizontal_scale=0.25,
                vertical_scale=0.01,
                subterrain_types=[
                    ["flat_terrain", "pyramid_sloped_terrain", "flat_terrain"],
                    ["pyramid_sloped_terrain", "pyramid_sloped_terrain", "pyramid_sloped_terrain"],
                    ["flat_terrain", "pyramid_sloped_terrain", "flat_terrain"],
                ],
                pyramid_sloped_terrain={
                    "slope": slopes[difficulty]
                }
            ),
        )
    
    return scene


def create_simple_terrain(terrain_type: str, difficulty: int):
    """シンプルな地形を作成（代替案）"""
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 3.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
        ),
        show_viewer=True,
    )
    
    # 高さフィールドを直接作成
    if terrain_type == "steps":
        # 段差の高さフィールドを作成
        heights = {1: 5, 2: 10, 3: 15}  # 高さ値
        height_field = np.zeros((40, 40))
        
        # 中央部分に段差を作成
        for i in range(10, 30):
            for j in range(10, 30):
                if (i + j) % 8 < 4:  # 段差パターン
                    height_field[i, j] = heights[difficulty]
                    
    else:  # slopes
        # 傾斜の高さフィールドを作成
        slopes = {1: 0.1, 2: 0.2, 3: 0.3}
        height_field = np.zeros((40, 40))
        
        # 傾斜を作成
        for i in range(40):
            for j in range(40):
                height_field[i, j] = slopes[difficulty] * i * 10
    
    # 高さフィールドから地形を作成
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=0.25,
            vertical_scale=0.01,
            height_field=height_field,
        ),
    )
    
    return scene


def evaluate_terrain(terrain_type: str, difficulty: int, max_steps: int = 1000, use_simple: bool = False):
    """地形評価を実行"""
    
    descriptions = {
        "steps": {1: "低い段差 (5cm)", 2: "中程度の段差 (10cm)", 3: "高い段差 (15cm)"},
        "slopes": {1: "緩やかな傾斜 (6度)", 2: "中程度の傾斜 (11度)", 3: "急な傾斜 (17度)"}
    }
    
    print(f"\n=== 地形評価: {descriptions[terrain_type][difficulty]} ===")
    
    try:
        # シーン作成
        if use_simple:
            print("シンプルな地形生成を使用します...")
            scene = create_simple_terrain(terrain_type, difficulty)
        else:
            scene = create_terrain(terrain_type, difficulty)
    except Exception as e:
        print(f"地形作成エラー: {e}")
        print("シンプルな地形生成で再試行します...")
        scene = create_simple_terrain(terrain_type, difficulty)
    
    # ロボット追加
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=[0.0, 0.0, 0.5],
            quat=[0.0, 0.0, 0.0, 1.0],
        ),
    )
    
    scene.build(n_envs=1)
    
    # 評価実行
    results = {
        "success": False,
        "distance": 0.0,
        "steps": 0,
        "max_height": 0.0,
    }
    
    print("評価を開始します...")
    print("ヒント: ESCキーで終了できます")
    
    for step in range(max_steps):
        # 基本的な前進制御
        target_actions = torch.zeros(12, device=gs.device)
        robot.control_dofs_position(target_actions, list(range(12)))
        
        scene.step()
        
        # ロボット状態取得
        pos = robot.get_pos()[0]
        distance = float(pos[0])
        height = float(pos[2])
        
        results["distance"] = distance
        results["steps"] = step + 1
        results["max_height"] = max(results["max_height"], height)
        
        # 成功条件: 5m前進
        if distance > 5.0:
            results["success"] = True
            print(f"✓ 成功! {distance:.2f}m 前進 ({step+1} ステップ)")
            break
            
        # 失敗条件: 転倒
        if height < 0.2:
            print(f"✗ 転倒により失敗 ({step+1} ステップ)")
            break
            
        if step % 200 == 0:
            print(f"  Step {step}: 位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print(f"最終結果: {results['distance']:.2f}m 前進\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="地形汎化性能評価ツール")
    parser.add_argument("--terrain", choices=["steps", "slopes"], required=True,
                       help="地形タイプ")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3], default=1,
                       help="難易度レベル")
    parser.add_argument("--all_levels", action="store_true",
                       help="全レベルで評価")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="最大ステップ数")
    parser.add_argument("--simple", action="store_true",
                       help="シンプルな地形生成を使用")
    
    args = parser.parse_args()
    
    # Genesis初期化
    gs.init(seed=42, backend=gs.gpu)
    
    if args.all_levels:
        # 全レベル評価
        results = {}
        for level in [1, 2, 3]:
            results[level] = evaluate_terrain(args.terrain, level, args.max_steps, args.simple)
        
        # 結果サマリー
        print("=== 評価結果サマリー ===")
        for level, result in results.items():
            status = "✓ 成功" if result["success"] else "✗ 失敗"
            print(f"レベル {level}: {status} | 距離: {result['distance']:.2f}m")
    
    else:
        # 単一レベル評価
        result = evaluate_terrain(args.terrain, args.difficulty, args.max_steps, args.simple)


if __name__ == "__main__":
    main()