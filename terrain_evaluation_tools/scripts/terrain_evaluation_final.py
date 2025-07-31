"""
地形汎化性能評価ツール（最終修正版）
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
        # 段差設定 - 正の値で段差を作成
        step_configs = {
            1: {"step_height": 0.05, "step_width": 1.0},   # 5cm
            2: {"step_height": 0.10, "step_width": 0.8},   # 10cm  
            3: {"step_height": 0.15, "step_width": 0.6},   # 15cm
        }
        
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
                subterrain_parameters={
                    "pyramid_stairs_terrain": step_configs[difficulty]
                }
            ),
        )
    
    elif terrain_type == "slopes":
        # 傾斜設定
        slope_configs = {
            1: {"slope": 0.1},   # ~6 degrees
            2: {"slope": 0.2},   # ~11 degrees
            3: {"slope": 0.3},   # ~17 degrees
        }
        
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
                subterrain_parameters={
                    "pyramid_sloped_terrain": slope_configs[difficulty]
                }
            ),
        )
    
    return scene


def create_height_field_terrain(terrain_type: str, difficulty: int):
    """高さフィールドを使って地形を作成（代替案）"""
    
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
    
    # 高さフィールドを直接作成（48x48グリッド）
    height_field = np.zeros((48, 48), dtype=np.float32)
    
    if terrain_type == "steps":
        # 段差の高さ値（スケールファクターを考慮）
        step_heights = {1: 50, 2: 100, 3: 150}  # vertical_scale=0.01なので実際は5cm, 10cm, 15cm
        current_height = step_heights[difficulty]
        
        # 段差パターンを作成
        step_width = 8  # グリッド単位での段差幅
        for i in range(48):
            step_level = i // step_width
            for j in range(48):
                if 12 <= j <= 36:  # 中央部分に段差を配置
                    height_field[i, j] = step_level * current_height
                    
    else:  # slopes
        # 傾斜の角度に対応する高さ変化
        slope_factors = {1: 20, 2: 40, 3: 60}  # 傾斜の強さ
        slope_factor = slope_factors[difficulty]
        
        # 傾斜パターンを作成
        for i in range(48):
            for j in range(48):
                if 12 <= j <= 36:  # 中央部分に傾斜を配置
                    height_field[i, j] = i * slope_factor / 10
    
    # 高さフィールドから地形を作成
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=0.25,
            vertical_scale=0.01,
            height_field=height_field,
        ),
    )
    
    return scene


def evaluate_terrain(terrain_type: str, difficulty: int, max_steps: int = 1000, use_height_field: bool = False):
    """地形評価を実行"""
    
    descriptions = {
        "steps": {1: "低い段差 (5cm)", 2: "中程度の段差 (10cm)", 3: "高い段差 (15cm)"},
        "slopes": {1: "緩やかな傾斜 (6度)", 2: "中程度の傾斜 (11度)", 3: "急な傾斜 (17度)"}
    }
    
    print(f"\n=== 地形評価: {descriptions[terrain_type][difficulty]} ===")
    
    try:
        # シーン作成
        if use_height_field:
            print("高さフィールド方式で地形を生成します...")
            scene = create_height_field_terrain(terrain_type, difficulty)
        else:
            print("サブテレイン方式で地形を生成します...")
            scene = create_terrain(terrain_type, difficulty)
    except Exception as e:
        print(f"地形作成エラー: {e}")
        print("高さフィールド方式で再試行します...")
        scene = create_height_field_terrain(terrain_type, difficulty)
    
    # ロボット追加
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=[0.0, 0.0, 0.5],
            quat=[0.0, 0.0, 0.0, 1.0],
        ),
    )
    
    scene.build(n_envs=1)
    print("シーン構築完了")
    
    # 評価実行
    results = {
        "success": False,
        "distance": 0.0,
        "steps": 0,
        "max_height": 0.0,
        "min_height": 1.0,
    }
    
    print("評価を開始します...")
    print("ヒント: Ctrl+Cで中断できます")
    
    try:
        for step in range(max_steps):
            # 基本的な前進制御（デフォルト姿勢を維持）
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
            results["min_height"] = min(results["min_height"], height)
            
            # 成功条件: 5m前進
            if distance > 5.0:
                results["success"] = True
                print(f"✓ 成功! {distance:.2f}m 前進 ({step+1} ステップ)")
                break
                
            # 失敗条件: 転倒
            if height < 0.2:
                print(f"✗ 転倒により失敗 ({step+1} ステップ)")
                break
                
            # 進行状況表示
            if step % 200 == 0 and step > 0:
                print(f"  Step {step}: 位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        else:
            print(f"最大ステップ数 {max_steps} に到達しました")
    
    except KeyboardInterrupt:
        print("\n評価が中断されました")
    
    print(f"最終結果: {results['distance']:.2f}m 前進")
    print(f"高さ範囲: {results['min_height']:.2f}m - {results['max_height']:.2f}m\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="地形汎化性能評価ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python terrain_evaluation_final.py --terrain steps --difficulty 2
  python terrain_evaluation_final.py --terrain slopes --all_levels
  python terrain_evaluation_final.py --terrain steps --difficulty 3 --height_field
        """
    )
    
    parser.add_argument("--terrain", choices=["steps", "slopes"], required=True,
                       help="地形タイプ")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3],
                       help="難易度レベル")
    parser.add_argument("--all_levels", action="store_true",
                       help="全レベルで評価")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="最大ステップ数")
    parser.add_argument("--height_field", action="store_true",
                       help="高さフィールド方式で地形生成")
    
    args = parser.parse_args()
    
    if not args.all_levels and args.difficulty is None:
        parser.error("--difficulty または --all_levels を指定してください")
    
    # Genesis初期化
    gs.init(seed=42, backend=gs.gpu)
    
    if args.all_levels:
        # 全レベル評価
        results = {}
        for level in [1, 2, 3]:
            results[level] = evaluate_terrain(args.terrain, level, args.max_steps, args.height_field)
            
            # レベル間で少し休憩
            if level < 3:
                print("次のレベルまで3秒待機...")
                time.sleep(3)
        
        # 結果サマリー
        print("="*50)
        print(f"{args.terrain.upper()} 地形評価結果サマリー")
        print("="*50)
        for level, result in results.items():
            status = "✓ 成功" if result["success"] else "✗ 失敗"
            print(f"レベル {level}: {status} | 距離: {result['distance']:.2f}m | ステップ: {result['steps']}")
        print("="*50)
    
    else:
        # 単一レベル評価
        result = evaluate_terrain(args.terrain, args.difficulty, args.max_steps, args.height_field)


if __name__ == "__main__":
    main()