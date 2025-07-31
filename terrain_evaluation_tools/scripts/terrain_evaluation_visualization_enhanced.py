"""
地形汎化性能評価ツール（地形可視化強化版）
地形が正しく表示されるように改善したバージョン
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import genesis as gs


def create_visible_terrain(terrain_type: str, difficulty: int):
    """より明確に見える地形を作成"""
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.0, -5.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.5),
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
    
    # より大きな高さフィールド作成（128x128）
    height_field = np.zeros((128, 128), dtype=np.float32)
    
    if terrain_type == "steps":
        # より明確な段差（メートル単位で直接指定）
        step_heights = {1: 5.0, 2: 10.0, 3: 15.0}  # cm単位
        step_height_value = step_heights[difficulty]
        step_width = 16  # より広い段差
        
        print(f"段差地形生成:")
        print(f"  段差高さ: {step_height_value}cm")
        print(f"  段差幅: {step_width * 0.1:.1f}m")
        
        for i in range(128):
            step_level = i // step_width
            for j in range(128):
                # 段差を作成（高さフィールドの値を大きく）
                height_field[i, j] = step_level * step_height_value
                
    else:  # slopes
        slope_angles = {1: 6, 2: 11, 3: 17}  # 度
        angle = slope_angles[difficulty]
        slope_factor = np.tan(np.radians(angle))
        
        print(f"傾斜地形生成:")
        print(f"  傾斜角度: {angle}度")
        print(f"  傾斜率: {slope_factor:.3f}")
        
        for i in range(128):
            for j in range(128):
                # 傾斜を作成（距離に応じて高さを増加）
                distance = i * 0.1  # メートル単位
                height_field[i, j] = distance * slope_factor * 100  # cm単位
    
    # デバッグ出力
    print(f"\n高さフィールド統計:")
    print(f"  最小値: {height_field.min():.1f}cm")
    print(f"  最大値: {height_field.max():.1f}cm")
    print(f"  平均値: {height_field.mean():.1f}cm")
    
    # 地形作成（より大きなスケール）
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=0.1,  # 各グリッドが0.1m = 10cm
            vertical_scale=0.01,   # 高さ単位が0.01m = 1cm
            height_field=height_field,
        ),
    )
    
    # カメラを追加（録画用）
    camera = scene.add_camera(
        res=(1920, 1080),
        pos=(8.0, -8.0, 6.0),
        lookat=(0.0, 0.0, 1.0),
        fov=60,
        GUI=False,
    )
    
    return scene, camera, height_field


class SimplePolicy(nn.Module):
    """シンプルなポリシーネットワーク"""
    
    def __init__(self, input_dim=45, output_dim=12):
        super(SimplePolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output_activation(self.fc4(x))
        return x


def load_policy_simple(policy_path: str):
    """学習済みポリシーを読み込み"""
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"ポリシーファイルが見つかりません: {policy_path}")
    
    try:
        checkpoint = torch.load(policy_path, map_location=gs.device)
        policy = SimplePolicy().to(gs.device)
        
        try:
            if 'actor' in checkpoint:
                actor_state = checkpoint['actor']
                policy.load_state_dict(actor_state, strict=False)
                print("✓ RSL-RL形式のポリシーを読み込みました")
            elif 'model_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✓ 標準PyTorch形式のポリシーを読み込みました")
            elif isinstance(checkpoint, dict) and any('weight' in k for k in checkpoint.keys()):
                policy.load_state_dict(checkpoint, strict=False)
                print("✓ 状態辞書形式のポリシーを読み込みました")
            else:
                print("⚠ ポリシー形式を特定できません、ランダム初期化を使用")
                
        except Exception as e:
            print(f"⚠ ポリシー重みの読み込みに失敗: {e}")
            print("⚠ ランダム初期化のポリシーを使用")
        
        policy.eval()
        return policy
        
    except Exception as e:
        raise RuntimeError(f"ポリシー読み込みエラー: {e}") from e


def create_dummy_observation(robot):
    """Go2用のダミー観測を作成"""
    try:
        pos = robot.get_pos()[0]
        quat = robot.get_quat()[0]
        vel = robot.get_vel()[0] if hasattr(robot, 'get_vel') else torch.zeros(3, device=gs.device)
        ang_vel = robot.get_ang()[0] if hasattr(robot, 'get_ang') else torch.zeros(3, device=gs.device)
        
        obs = torch.cat([
            ang_vel,
            torch.tensor([0, 0, -1], device=gs.device),
            torch.zeros(3, device=gs.device),
            torch.zeros(12, device=gs.device),
            torch.zeros(12, device=gs.device),
            torch.zeros(12, device=gs.device),
        ])
        return obs
        
    except Exception as e:
        return torch.zeros(45, device=gs.device)


def visualize_terrain_enhanced(terrain_type: str, difficulty: int, 
                              policy_path: str = None, max_steps: int = 1000,
                              save_video: bool = False):
    """地形を強化された可視化で確認"""
    
    descriptions = {
        "steps": {1: "低い段差 (5cm)", 2: "中程度の段差 (10cm)", 3: "高い段差 (15cm)"},
        "slopes": {1: "緩やかな傾斜 (6度)", 2: "中程度の傾斜 (11度)", 3: "急な傾斜 (17度)"}
    }
    
    print(f"\n=== 地形可視化（強化版）: {descriptions[terrain_type][difficulty]} ===")
    
    # シーン作成
    scene, camera, height_field = create_visible_terrain(terrain_type, difficulty)
    
    # ロボット追加（地形の高さに応じて配置）
    initial_height = height_field[0, 64] * 0.01 + 0.5  # 地形の中央の高さ + マージン
    print(f"ロボット初期高さ: {initial_height:.2f}m")
    
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=[0.0, 0.0, initial_height],
            quat=[0.0, 0.0, 0.0, 1.0],
        ),
    )
    
    # サイズ参照用のマーカー追加
    # 10cmの立方体を複数配置
    for i in range(5):
        marker = scene.add_entity(
            gs.morphs.Box(
                pos=(2.0, i * 0.5 - 1.0, height_field[20, 64] * 0.01 + 0.05),
                size=(0.1, 0.1, 0.1),  # 10cm立方体
            ),
        )
    
    print("シーンを構築中...")
    scene.build(n_envs=1)
    print("✓ シーン構築完了")
    
    # Go2の関節設定
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ]
    
    try:
        motor_dof_idx = [robot.get_joint(name).dof_start for name in joint_names]
        robot.set_dofs_kp([20.0] * 12, motor_dof_idx)
        robot.set_dofs_kv([0.5] * 12, motor_dof_idx)
        print("✓ PD制御パラメータを設定")
    except Exception as e:
        print(f"⚠ 関節設定エラー: {e}")
        motor_dof_idx = list(range(12))
    
    # ポリシー読み込み
    if policy_path:
        try:
            policy = load_policy_simple(policy_path)
            print("✓ 学習済みポリシーを使用")
            use_policy = True
        except Exception as e:
            print(f"⚠ ポリシー読み込み失敗: {e}")
            use_policy = False
    else:
        use_policy = False
    
    # 動画記録
    video_path = None
    if save_video:
        video_dir = "terrain_visualization"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/terrain_{terrain_type}_level{difficulty}_enhanced_{int(time.time())}.mp4"
        print(f"📹 動画記録開始: {video_path}")
        camera.start_recording()
    
    # カメラを移動させて地形全体を確認
    print("\n地形全体を確認中...")
    radius = 10.0
    height = 5.0
    
    for i in range(max_steps):
        # カメラ位置を円周上で移動
        angle = i * 2 * np.pi / 360  # 1度ずつ回転
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_height = height + 2 * np.sin(i * 0.05)  # 高さも変化
        
        # ビューワーのカメラを更新
        scene.viewer.set_camera_pose(
            camera_pos=(cam_x, cam_y, cam_height),
            camera_lookat=(0.0, 0.0, 1.0),
        )
        
        # ポリシーを使用する場合
        if use_policy and i > 100:  # 100ステップ後から動作開始
            try:
                obs = create_dummy_observation(robot)
                with torch.no_grad():
                    if obs.dim() == 1:
                        obs = obs.unsqueeze(0)
                    actions = policy(obs)
                    if actions.dim() > 1:
                        actions = actions.squeeze(0)
                actions = torch.clamp(actions, -1.0, 1.0)
                robot.control_dofs_position(actions, motor_dof_idx)
            except:
                pass
        
        scene.step()
        
        if save_video:
            camera.render()
        
        # 進行状況
        if i % 100 == 0:
            pos = robot.get_pos()[0]
            print(f"  ステップ {i}: カメラ角度 {i % 360}度, ロボット位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # 動画保存
    if save_video:
        print("📹 動画記録停止中...")
        camera.stop_recording(save_to_filename=video_path, fps=30)
        print(f"✅ 動画保存完了: {video_path}")
    
    print("\n地形詳細:")
    print(f"- 地形タイプ: {terrain_type}")
    print(f"- 難易度: {difficulty}")
    print(f"- グリッドサイズ: 128x128")
    print(f"- 水平スケール: 0.1m/グリッド (10cm)")
    print(f"- 垂直スケール: 0.01m/単位 (1cm)")
    print(f"- 地形サイズ: {128*0.1:.1f}m x {128*0.1:.1f}m")
    print(f"- 最大高さ: {height_field.max() * 0.01:.2f}m")
    
    return video_path


def main():
    parser = argparse.ArgumentParser(
        description="地形可視化強化ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 段差地形を確認
  python terrain_evaluation_visualization_enhanced.py --terrain steps --difficulty 2
  
  # 学習済みポリシーで動作確認
  python terrain_evaluation_visualization_enhanced.py --terrain steps --difficulty 2 \\
      --policy logs/go2-walking/model_100.pt
  
  # 動画で記録
  python terrain_evaluation_visualization_enhanced.py --terrain steps --difficulty 2 --save_video
  
  # 全レベルを確認
  python terrain_evaluation_visualization_enhanced.py --terrain steps --all_levels
        """
    )
    
    parser.add_argument("--terrain", choices=["steps", "slopes"], required=True,
                       help="地形タイプ")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3],
                       help="難易度レベル")
    parser.add_argument("--all_levels", action="store_true",
                       help="全レベルを確認")
    parser.add_argument("--policy", type=str,
                       help="学習済みポリシーファイル(.pt)")
    parser.add_argument("--save_video", action="store_true",
                       help="動画を記録する")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="最大ステップ数")
    
    args = parser.parse_args()
    
    if not args.all_levels and args.difficulty is None:
        parser.error("--difficulty または --all_levels を指定してください")
    
    # Genesis初期化
    gs.init(seed=42, backend=gs.gpu)
    
    if args.all_levels:
        # 全レベル確認
        for level in [1, 2, 3]:
            visualize_terrain_enhanced(args.terrain, level, args.policy, 
                                     args.max_steps, args.save_video)
            if level < 3:
                print("\n" + "="*60 + "\n")
    else:
        # 単一レベル確認
        visualize_terrain_enhanced(args.terrain, args.difficulty, args.policy,
                                 args.max_steps, args.save_video)


if __name__ == "__main__":
    main()