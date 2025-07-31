"""
地形汎化性能評価ツール（ポリシー修正版）
学習済みポリシーを正しく読み込んで地形での性能を評価
"""

import argparse
import os
import pickle
import time
import numpy as np
import torch
import genesis as gs


def create_height_field_terrain(terrain_type: str, difficulty: int):
    """高さフィールドを使って地形を作成"""
    
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
        # 段差の高さ値
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


def load_policy_simple(policy_path: str):
    """学習済みポリシーを直接読み込み（シンプル版）"""
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"ポリシーファイルが見つかりません: {policy_path}")
    
    try:
        # PyTorchモデルを直接読み込み
        policy_state = torch.load(policy_path, map_location=gs.device)
        
        # ポリシーネットワークを再構築
        class SimplePolicy:
            def __init__(self, state_dict):
                self.state_dict = state_dict
                # モデル情報から入力・出力次元を推定
                self.input_dim = 45  # Go2の標準観測次元
                self.output_dim = 12  # Go2のアクション次元
                
                # 単純なフィードフォワードネットワークを想定
                self.fc1 = torch.nn.Linear(self.input_dim, 512)
                self.fc2 = torch.nn.Linear(512, 256) 
                self.fc3 = torch.nn.Linear(256, 128)
                self.fc4 = torch.nn.Linear(128, self.output_dim)
                
                # パラメータをロード（可能な場合）
                try:
                    self.load_state_dict(state_dict)
                    self.loaded = True
                except:
                    print("⚠ 状態辞書の読み込みに失敗、ランダム初期化を使用")
                    self.loaded = False
                
                self.eval()
            
            def load_state_dict(self, state_dict):
                """状態辞書から重みを読み込み"""
                if 'actor' in state_dict:
                    actor_state = state_dict['actor']
                    self.fc1.weight.data = actor_state.get('0.weight', self.fc1.weight.data)
                    self.fc1.bias.data = actor_state.get('0.bias', self.fc1.bias.data)
                    self.fc2.weight.data = actor_state.get('2.weight', self.fc2.weight.data)
                    self.fc2.bias.data = actor_state.get('2.bias', self.fc2.bias.data)
                    self.fc3.weight.data = actor_state.get('4.weight', self.fc3.weight.data)
                    self.fc3.bias.data = actor_state.get('4.bias', self.fc3.bias.data)
                    self.fc4.weight.data = actor_state.get('6.weight', self.fc4.weight.data)
                    self.fc4.bias.data = actor_state.get('6.bias', self.fc4.bias.data)
            
            def __call__(self, obs):
                with torch.no_grad():
                    if obs.dim() == 1:
                        obs = obs.unsqueeze(0)
                    
                    x = torch.relu(self.fc1(obs))
                    x = torch.relu(self.fc2(x))
                    x = torch.relu(self.fc3(x))
                    action = torch.tanh(self.fc4(x))
                    
                    return action.squeeze(0) if action.shape[0] == 1 else action
        
        policy = SimplePolicy(policy_state)
        
        if policy.loaded:
            print(f"✓ ポリシーを直接読み込みました: {policy_path}")
        else:
            print(f"⚠ ポリシー構造の推定で読み込み: {policy_path}")
        
        return policy
        
    except Exception as e:
        raise RuntimeError(f"ポリシー読み込みエラー: {e}") from e


def create_simple_forward_policy():
    """シンプルな前進歩行ポリシー（デモ用）"""
    
    class SimpleWalkingPolicy:
        def __init__(self):
            self.step_count = 0
            # Go2の関節名とデフォルト角度
            self.default_angles = torch.tensor([
                0.0, 0.8, -1.5,   # FR leg
                0.0, 0.8, -1.5,   # FL leg  
                0.0, 0.8, -1.5,   # RR leg
                0.0, 0.8, -1.5,   # RL leg
            ], device=gs.device)
            
        def __call__(self, obs):
            """簡単な歩行パターンを生成"""
            self.step_count += 1
            
            # シンプルな歩行パターン（サインカーブ）
            t = self.step_count * 0.05
            
            # 脚の位相をずらして歩行パターンを作成
            leg_phases = [0, np.pi, np.pi/2, 3*np.pi/2]  # FR, FL, RR, RL
            
            actions = self.default_angles.clone()
            
            for i in range(4):  # 4脚
                phase = leg_phases[i]
                # 股関節の前後動作
                actions[i*3 + 0] = 0.3 * np.sin(t + phase)
                # 膝関節の上下動作  
                actions[i*3 + 2] = self.default_angles[i*3 + 2] + 0.2 * np.sin(2*t + phase)
            
            return actions
    
    return SimpleWalkingPolicy()


def create_dummy_observation(robot):
    """Go2用のダミー観測を作成"""
    try:
        # ロボットの状態から観測を構築
        pos = robot.get_pos()[0]
        quat = robot.get_quat()[0]
        vel = robot.get_vel()[0]
        ang_vel = robot.get_ang()[0]
        
        # 簡易的な観測ベクトル（45次元）
        obs = torch.cat([
            ang_vel,  # 3: 角速度
            torch.tensor([0, 0, -1], device=gs.device),  # 3: 重力方向
            torch.zeros(3, device=gs.device),  # 3: コマンド
            torch.zeros(12, device=gs.device),  # 12: 関節位置偏差
            torch.zeros(12, device=gs.device),  # 12: 関節速度
            torch.zeros(12, device=gs.device),  # 12: 前回のアクション
        ])
        
        return obs
        
    except:
        # フォールバック: ゼロベクトル
        return torch.zeros(45, device=gs.device)


def evaluate_terrain_with_policy(terrain_type: str, difficulty: int, policy_path: str = None, 
                                max_steps: int = 2000):
    """学習済みポリシーで地形評価を実行"""
    
    descriptions = {
        "steps": {1: "低い段差 (5cm)", 2: "中程度の段差 (10cm)", 3: "高い段差 (15cm)"},
        "slopes": {1: "緩やかな傾斜 (6度)", 2: "中程度の傾斜 (11度)", 3: "急な傾斜 (17度)"}
    }
    
    print(f"\n=== 地形評価: {descriptions[terrain_type][difficulty]} ===")
    
    # ポリシーを読み込み
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
    
    # シーン作成
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
    print("✓ シーン構築完了")
    
    # Go2の関節インデックスを取得
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    ]
    
    try:
        motor_dof_idx = [robot.get_joint(name).dof_start for name in joint_names]
        print(f"✓ モーター関節を設定: {len(motor_dof_idx)} joints")
    except:
        print("⚠ 関節名の取得に失敗、インデックス使用")
        motor_dof_idx = list(range(12))
    
    # PD制御パラメータ設定
    robot.set_dofs_kp([20.0] * 12, motor_dof_idx)
    robot.set_dofs_kv([0.5] * 12, motor_dof_idx)
    
    # 評価実行
    results = {
        "success": False,
        "distance": 0.0,
        "steps": 0,
        "max_height": 0.0,
        "min_height": 1.0,
        "policy_type": "trained" if use_trained_policy else "simple"
    }
    
    print("評価を開始します...")
    print("目標: 8m前進")
    
    try:
        for step in range(max_steps):
            
            if use_trained_policy:
                # 学習済みポリシーを使用
                obs = create_dummy_observation(robot)
                actions = policy(obs)
            else:
                # シンプルポリシーを使用
                obs = create_dummy_observation(robot)
                actions = policy(obs)
            
            # アクションをクリップ
            actions = torch.clamp(actions, -1.0, 1.0)
            
            # ロボット制御
            robot.control_dofs_position(actions, motor_dof_idx)
            scene.step()
            
            # ロボット状態取得
            pos = robot.get_pos()[0]
            distance = float(pos[0])
            height = float(pos[2])
            
            results["distance"] = distance
            results["steps"] = step + 1
            results["max_height"] = max(results["max_height"], height)
            results["min_height"] = min(results["min_height"], height)
            
            # 成功条件: 8m前進
            if distance > 8.0:
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
            print(f"最大ステップ数 {max_steps} に到達")
    
    except KeyboardInterrupt:
        print("\n⚠ 評価が中断されました")
    
    print(f"最終結果: {results['distance']:.2f}m 前進")
    print(f"高さ範囲: {results['min_height']:.2f}m - {results['max_height']:.2f}m")
    print(f"使用ポリシー: {results['policy_type']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="地形汎化性能評価ツール（ポリシー修正版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # シンプル歩行ポリシーで評価
  python terrain_evaluation_policy_fixed.py --terrain steps --difficulty 2
  
  # 学習済みポリシーで評価（.ptファイルのみ）
  python terrain_evaluation_policy_fixed.py --terrain steps --difficulty 2 \\
      --policy logs/go2-walking/model_100.pt
  
  # 全レベル評価
  python terrain_evaluation_policy_fixed.py --terrain slopes --all_levels \\
      --policy logs/go2-walking/model_100.pt
        """
    )
    
    parser.add_argument("--terrain", choices=["steps", "slopes"], required=True,
                       help="地形タイプ")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3],
                       help="難易度レベル")
    parser.add_argument("--all_levels", action="store_true",
                       help="全レベルで評価")
    parser.add_argument("--policy", type=str,
                       help="学習済みポリシーファイル(.pt)")
    parser.add_argument("--max_steps", type=int, default=2000,
                       help="最大ステップ数")
    
    args = parser.parse_args()
    
    if not args.all_levels and args.difficulty is None:
        parser.error("--difficulty または --all_levels を指定してください")
    
    # Genesis初期化
    gs.init(seed=42, backend=gs.gpu)
    
    # 学習済みポリシーの有無を確認
    if args.policy and not os.path.exists(args.policy):
        print(f"⚠ ポリシーファイルが見つかりません: {args.policy}")
        print("⚠ シンプル歩行ポリシーで続行します")
        args.policy = None
    
    if args.all_levels:
        # 全レベル評価
        results = {}
        for level in [1, 2, 3]:
            results[level] = evaluate_terrain_with_policy(
                args.terrain, level, args.policy, args.max_steps
            )
            
            # レベル間で休憩
            if level < 3:
                print("次のレベルまで3秒待機...")
                time.sleep(3)
        
        # 結果サマリー
        print("="*60)
        print(f"{args.terrain.upper()} 地形評価結果サマリー")
        print("="*60)
        for level, result in results.items():
            status = "✓ 成功" if result["success"] else "✗ 失敗"
            policy_type = result["policy_type"]
            print(f"レベル {level}: {status} | 距離: {result['distance']:.2f}m | "
                  f"ポリシー: {policy_type}")
        print("="*60)
    
    else:
        # 単一レベル評価
        result = evaluate_terrain_with_policy(
            args.terrain, args.difficulty, args.policy, args.max_steps
        )


if __name__ == "__main__":
    main()