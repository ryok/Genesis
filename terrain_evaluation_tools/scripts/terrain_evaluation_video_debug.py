"""
地形汎化性能評価ツール（デバッグ版）
地形が正しく生成されているか確認するためのツール
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import genesis as gs


def create_debug_terrain(terrain_type: str, difficulty: int):
    """地形をデバッグ用に作成（より明確な可視化）"""
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, -6.0, 4.0),  # より遠くから全体を見る
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=60,
            max_FPS=50,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
        ),
        show_viewer=True,
    )
    
    # 高さフィールド作成（より大きな変化）
    height_field = np.zeros((64, 64), dtype=np.float32)
    
    if terrain_type == "steps":
        # レベル2で10cmの段差 = 100 * 0.01 = 1.0
        step_heights = {1: 50, 2: 100, 3: 150}  # vertical_scale=0.01での値
        current_height = step_heights[difficulty]
        step_width = 8
        
        print(f"段差生成: 高さ={current_height * 0.01:.2f}m, 幅={step_width * 0.25:.2f}m")
        
        # より明確な段差パターン
        for i in range(64):
            step_level = i // step_width
            for j in range(64):
                # 全幅で段差を作成（より見やすく）
                height_field[i, j] = step_level * current_height
                
    else:  # slopes
        slope_factors = {1: 30, 2: 60, 3: 90}  # より急な傾斜
        slope_factor = slope_factors[difficulty]
        
        print(f"傾斜生成: 傾斜率={slope_factor}")
        
        for i in range(64):
            for j in range(64):
                # 全幅で傾斜を作成
                height_field[i, j] = i * slope_factor / 10
    
    # デバッグ出力
    print(f"高さフィールド統計:")
    print(f"  最小値: {height_field.min():.3f}")
    print(f"  最大値: {height_field.max():.3f}")
    print(f"  平均値: {height_field.mean():.3f}")
    
    # 地形作成
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=0.25,  # 各グリッドが0.25m
            vertical_scale=0.01,    # 高さ単位が0.01m
            height_field=height_field,
        ),
    )
    
    # 平面を追加（比較用）
    plane = scene.add_entity(
        gs.morphs.Plane(
            pos=(0, 8, 0),  # 横に配置
            normal=(0, 0, 1),
            size=(10, 10),
        ),
    )
    
    # カメラを追加（録画用）
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(8.0, -8.0, 6.0),  # 斜め上から全体を見下ろす
        lookat=(0.0, 0.0, 1.0),
        fov=60,
        GUI=False,
    )
    
    return scene, camera


def visualize_terrain(terrain_type: str, difficulty: int, save_video: bool = False):
    """地形を可視化して確認"""
    
    descriptions = {
        "steps": {1: "低い段差 (5cm)", 2: "中程度の段差 (10cm)", 3: "高い段差 (15cm)"},
        "slopes": {1: "緩やかな傾斜 (6度)", 2: "中程度の傾斜 (11度)", 3: "急な傾斜 (17度)"}
    }
    
    print(f"\n=== 地形デバッグ: {descriptions[terrain_type][difficulty]} ===")
    
    # シーン作成
    scene, camera = create_debug_terrain(terrain_type, difficulty)
    
    # ロボット追加（スケール確認用）
    print("ロボットを追加中...")
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=[0.0, 0.0, 1.0],  # 地形の上に配置
            quat=[0.0, 0.0, 0.0, 1.0],
        ),
    )
    
    # ボックスを追加（サイズ参照用）
    reference_box = scene.add_entity(
        gs.morphs.Box(
            pos=(2.0, 0.0, 0.5),
            size=(0.1, 0.1, 1.0),  # 10cm x 10cm x 1m
        ),
    )
    
    print("シーンを構築中...")
    scene.build(n_envs=1)
    print("✓ シーン構築完了")
    
    # 動画記録
    video_path = None
    if save_video:
        video_dir = "debug_videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/terrain_{terrain_type}_level{difficulty}_{int(time.time())}.mp4"
        print(f"📹 動画記録開始: {video_path}")
        camera.start_recording()
    
    # カメラを回転させて地形を確認
    print("地形を様々な角度から確認中...")
    radius = 10.0
    height = 6.0
    
    for i in range(360):
        # カメラ位置を円周上で移動
        angle = i * np.pi / 180
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        
        # ビューワーのカメラを更新
        scene.viewer.set_camera_pose(
            camera_pos=(cam_x, cam_y, height),
            camera_lookat=(0.0, 0.0, 1.0),
        )
        
        scene.step()
        
        if save_video:
            camera.render()
        
        # 進行状況
        if i % 60 == 0:
            print(f"  回転角度: {i}度")
    
    # 動画保存
    if save_video:
        print("📹 動画記録停止中...")
        camera.stop_recording(save_to_filename=video_path, fps=30)
        print(f"✅ 動画保存完了: {video_path}")
    
    print("\n地形統計:")
    print(f"- 地形タイプ: {terrain_type}")
    print(f"- 難易度: {difficulty}")
    print(f"- グリッドサイズ: 64x64")
    print(f"- 水平スケール: 0.25m/グリッド")
    print(f"- 垂直スケール: 0.01m/単位")
    print(f"- 地形サイズ: {64*0.25:.1f}m x {64*0.25:.1f}m")
    
    return video_path


def main():
    parser = argparse.ArgumentParser(
        description="地形生成デバッグツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 段差地形を確認
  python terrain_evaluation_video_debug.py --terrain steps --difficulty 2
  
  # 動画で記録
  python terrain_evaluation_video_debug.py --terrain steps --difficulty 2 --save_video
  
  # 全レベルを確認
  python terrain_evaluation_video_debug.py --terrain steps --all_levels
        """
    )
    
    parser.add_argument("--terrain", choices=["steps", "slopes"], required=True,
                       help="地形タイプ")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3],
                       help="難易度レベル")
    parser.add_argument("--all_levels", action="store_true",
                       help="全レベルを確認")
    parser.add_argument("--save_video", action="store_true",
                       help="動画を記録する")
    
    args = parser.parse_args()
    
    if not args.all_levels and args.difficulty is None:
        parser.error("--difficulty または --all_levels を指定してください")
    
    # Genesis初期化
    gs.init(seed=42, backend=gs.gpu)
    
    if args.all_levels:
        # 全レベル確認
        for level in [1, 2, 3]:
            visualize_terrain(args.terrain, level, args.save_video)
            if level < 3:
                print("\n" + "="*60 + "\n")
    else:
        # 単一レベル確認
        visualize_terrain(args.terrain, args.difficulty, args.save_video)


if __name__ == "__main__":
    main()