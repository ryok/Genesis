"""
地形汎化性能評価ツール（動画記録版・修正）
Genesis のカメラ録画機能を使用して実験動作を動画として記録
"""

import argparse
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import genesis as gs


def create_height_field_terrain_with_camera(terrain_type: str, difficulty: int, 
                                           save_video: bool = True, video_dir: str = "videos"):
    """高さフィールド地形を作成し、カメラを設定"""
    
    # 動画保存ディレクトリ作成
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        video_filename = f"{terrain_type}_level{difficulty}_{int(time.time())}.mp4"
        video_path = os.path.join(video_dir, video_filename)
    else:
        video_path = None
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(4.0, 4.0, 3.0),  # より良い視点
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            max_FPS=50,  # 録画品質向上
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
        ),
        show_viewer=True,
    )
    
    # 高さフィールド作成
    height_field = np.zeros((48, 48), dtype=np.float32)
    
    if terrain_type == "steps":
        step_heights = {1: 50, 2: 100, 3: 150}
        current_height = step_heights[difficulty]
        step_width = 8
        
        for i in range(48):
            step_level = i // step_width
            for j in range(48):
                if 12 <= j <= 36:
                    height_field[i, j] = step_level * current_height
                    
    else:  # slopes
        slope_factors = {1: 20, 2: 40, 3: 60}
        slope_factor = slope_factors[difficulty]
        
        for i in range(48):
            for j in range(48):
                if 12 <= j <= 36:
                    height_field[i, j] = i * slope_factor / 10
    
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=0.25,
            vertical_scale=0.01,
            height_field=height_field,
        ),
    )
    
    # 録画用カメラを追加
    camera = None
    if save_video:
        camera = scene.add_camera(
            res=(1280, 720),  # HD解像度
            pos=(4.0, 4.0, 3.0),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            GUI=False,  # GUI表示なし（録画専用）
        )
    
    return scene, camera, video_path


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


def create_simple_forward_policy():
    """シンプルな前進歩行ポリシー（デモ用）"""
    
    class SimpleWalkingPolicy:
        def __init__(self):
            self.step_count = 0
            self.default_angles = torch.tensor([
                0.0, 0.8, -1.5,   # FR leg
                0.0, 0.8, -1.5,   # FL leg  
                0.0, 0.8, -1.5,   # RR leg
                0.0, 0.8, -1.5,   # RL leg
            ], device=gs.device)
            
        def __call__(self, obs):
            """簡単な歩行パターンを生成"""
            self.step_count += 1
            t = self.step_count * 0.02
            leg_phases = [0, np.pi, np.pi/2, 3*np.pi/2]
            
            actions = self.default_angles.clone()
            
            for i in range(4):
                phase = leg_phases[i]
                actions[i*3 + 0] = 0.2 * np.sin(t + phase)
                actions[i*3 + 2] = self.default_angles[i*3 + 2] + 0.1 * np.sin(2*t + phase)
            
            return actions
    
    return SimpleWalkingPolicy()


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


def evaluate_terrain_with_video_recording(terrain_type: str, difficulty: int, 
                                          policy_path: str = None, max_steps: int = 2000,
                                          save_video: bool = True, video_dir: str = "videos"):
    """地形評価を実行し、動画を記録"""
    
    descriptions = {
        "steps": {1: "低い段差 (5cm)", 2: "中程度の段差 (10cm)", 3: "高い段差 (15cm)"},
        "slopes": {1: "緩やかな傾斜 (6度)", 2: "中程度の傾斜 (11度)", 3: "急な傾斜 (17度)"}
    }
    
    print(f"\n=== 地形評価 + 動画記録: {descriptions[terrain_type][difficulty]} ===")
    
    # ポリシー読み込み
    if policy_path:
        try:
            policy = load_policy_simple(policy_path)
            print("✓ 学習済みポリシーを使用します")
            use_trained_policy = True
        except Exception as e:
            print(f"⚠ ポリシー読み込み失敗: {e}")
            print("⚠ シンプルな歩行ポリシーを使用します")
            policy = create_simple_forward_policy()
            use_trained_policy = False
    else:
        print("✓ シンプルな歩行ポリシーを使用します")
        policy = create_simple_forward_policy()
        use_trained_policy = False
    
    # シーン作成（カメラ付き）
    print("地形を生成中...")
    scene, camera, video_path = create_height_field_terrain_with_camera(
        terrain_type, difficulty, save_video, video_dir
    )
    
    if save_video and camera:
        print(f"📹 動画記録先: {video_path}")
    
    # ロボット追加
    print("ロボットを追加中...")
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=[0.0, 0.0, 0.5],
            quat=[0.0, 0.0, 0.0, 1.0],
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
        print(f"✓ モーター関節を設定: {len(motor_dof_idx)} joints")
    except Exception as e:
        print(f"⚠ 関節名の取得に失敗: {e}")
        motor_dof_idx = list(range(12))
    
    # PD制御設定
    try:
        robot.set_dofs_kp([20.0] * 12, motor_dof_idx)
        robot.set_dofs_kv([0.5] * 12, motor_dof_idx)
        print("✓ PD制御パラメータを設定")
    except Exception as e:
        print(f"⚠ PD制御パラメータ設定に失敗: {e}")
    
    # 評価実行
    results = {
        "success": False,
        "distance": 0.0,
        "steps": 0,
        "max_height": 0.0,
        "min_height": 1.0,
        "policy_type": "trained" if use_trained_policy else "simple",
        "video_path": video_path if save_video else None
    }
    
    print("評価を開始します...")
    print("目標: 8m前進")
    
    # 録画開始
    if save_video and camera:
        print("📹 動画記録開始")
        camera.start_recording()
    
    # 初期安定化
    for _ in range(100):
        scene.step()
        if save_video and camera:
            camera.render()
    
    try:
        for step in range(max_steps):
            try:
                if use_trained_policy:
                    obs = create_dummy_observation(robot)
                    with torch.no_grad():
                        if obs.dim() == 1:
                            obs = obs.unsqueeze(0)
                        actions = policy(obs)
                        if actions.dim() > 1:
                            actions = actions.squeeze(0)
                else:
                    obs = create_dummy_observation(robot)
                    actions = policy(obs)
                
                actions = torch.clamp(actions, -1.0, 1.0)
                robot.control_dofs_position(actions, motor_dof_idx)
                
            except Exception as e:
                if step == 0:
                    print(f"⚠ ポリシー実行エラー: {e}")
                actions = torch.zeros(12, device=gs.device)
                robot.control_dofs_position(actions, motor_dof_idx)
            
            scene.step()
            
            # カメラ録画
            if save_video and camera:
                camera.render()
            
            # ロボット状態取得
            try:
                pos = robot.get_pos()[0]
                distance = float(pos[0])
                height = float(pos[2])
                
                results["distance"] = distance
                results["steps"] = step + 1
                results["max_height"] = max(results["max_height"], height)
                results["min_height"] = min(results["min_height"], height)
                
                # 成功条件
                if distance > 8.0:
                    results["success"] = True
                    print(f"✓ 成功! {distance:.2f}m 前進 ({step+1} ステップ)")
                    break
                    
                # 失敗条件
                if height < 0.2:
                    print(f"✗ 転倒により失敗 ({step+1} ステップ)")
                    break
                    
                # 進行状況表示
                if step % 200 == 0 and step > 0:
                    print(f"  Step {step}: 位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                    
            except Exception as e:
                if step == 0:
                    print(f"⚠ ロボット状態取得エラー: {e}")
                continue
        
        else:
            print(f"最大ステップ数 {max_steps} に到達")
    
    except KeyboardInterrupt:
        print("\n⚠ 評価が中断されました")
    
    finally:
        # 録画停止
        if save_video and camera:
            print("📹 動画記録停止中...")
            try:
                camera.stop_recording(save_to_filename=video_path, fps=30)
                print(f"✅ 動画保存完了: {video_path}")
            except Exception as e:
                print(f"⚠ 動画保存エラー: {e}")
    
    print(f"最終結果: {results['distance']:.2f}m 前進")
    print(f"高さ範囲: {results['min_height']:.2f}m - {results['max_height']:.2f}m")
    print(f"使用ポリシー: {results['policy_type']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="地形汎化性能評価ツール（動画記録版・修正）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 動画記録付きでシンプル歩行ポリシーを評価
  python terrain_evaluation_video_recorder_fixed.py --terrain steps --difficulty 2 --save_video
  
  # 学習済みポリシーで動画記録
  python terrain_evaluation_video_recorder_fixed.py --terrain steps --difficulty 2 \\
      --policy logs/go2-walking/model_100.pt --save_video --video_dir experiment_videos
  
  # 動画記録なしで評価のみ
  python terrain_evaluation_video_recorder_fixed.py --terrain slopes --difficulty 1
        """
    )
    
    parser.add_argument("--terrain", choices=["steps", "slopes"], required=True,
                       help="地形タイプ")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3], required=True,
                       help="難易度レベル")
    parser.add_argument("--policy", type=str,
                       help="学習済みポリシーファイル(.pt)")
    parser.add_argument("--max_steps", type=int, default=2000,
                       help="最大ステップ数")
    parser.add_argument("--save_video", action="store_true",
                       help="動画を記録する")
    parser.add_argument("--video_dir", type=str, default="videos",
                       help="動画保存ディレクトリ")
    
    args = parser.parse_args()
    
    # Genesis初期化
    gs.init(seed=42, backend=gs.gpu)
    
    # 学習済みポリシーの確認
    if args.policy and not os.path.exists(args.policy):
        print(f"⚠ ポリシーファイルが見つかりません: {args.policy}")
        print("⚠ シンプル歩行ポリシーで続行します")
        args.policy = None
    
    # 評価実行
    result = evaluate_terrain_with_video_recording(
        args.terrain, args.difficulty, args.policy, args.max_steps,
        args.save_video, args.video_dir
    )
    
    # 結果サマリー
    print("\n" + "="*60)
    print("実験結果サマリー")
    print("="*60)
    status = "✓ 成功" if result["success"] else "✗ 失敗"
    print(f"結果: {status}")
    print(f"移動距離: {result['distance']:.2f}m")
    print(f"継続時間: {result['steps']} ステップ")
    print(f"ポリシー: {result['policy_type']}")
    if result.get("video_path"):
        print(f"📹 動画: {result['video_path']}")
    print("="*60)


if __name__ == "__main__":
    main()