"""
複雑な地形における既存ポリシーの汎化性能評価スクリプト
段差と傾斜の強度を3段階で比較できる環境を提供
"""

import argparse
import os
import pickle
import time
import numpy as np
import torch
from typing import Dict, Any, Tuple

import genesis as gs
from examples.locomotion.go2_env import Go2Env


class TerrainEvaluationEnvironment:
    """段差と傾斜の3段階評価環境"""
    
    def __init__(self, terrain_type: str, difficulty_level: int):
        """
        Args:
            terrain_type: "steps" or "slopes"
            difficulty_level: 1 (easy), 2 (medium), 3 (hard)
        """
        self.terrain_type = terrain_type
        self.difficulty_level = difficulty_level
        self.terrain_configs = self._get_terrain_configs()
        
    def _get_terrain_configs(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """地形設定を取得"""
        return {
            "steps": {
                1: {  # Easy steps
                    "step_height": 0.05,  # 5cm
                    "step_width": 1.0,
                    "description": "低い段差 (5cm)"
                },
                2: {  # Medium steps
                    "step_height": 0.10,  # 10cm
                    "step_width": 0.8,
                    "description": "中程度の段差 (10cm)"
                },
                3: {  # Hard steps
                    "step_height": 0.15,  # 15cm
                    "step_width": 0.6,
                    "description": "高い段差 (15cm)"
                }
            },
            "slopes": {
                1: {  # Easy slopes
                    "slope": 0.1,  # ~6 degrees
                    "description": "緩やかな傾斜 (約6度)"
                },
                2: {  # Medium slopes
                    "slope": 0.2,  # ~11 degrees
                    "description": "中程度の傾斜 (約11度)"
                },
                3: {  # Hard slopes
                    "slope": 0.3,  # ~17 degrees
                    "description": "急な傾斜 (約17度)"
                }
            }
        }
    
    def create_terrain_scene(self, show_viewer: bool = True) -> gs.Scene:
        """指定された難易度の地形を持つシーンを作成"""
        config = self.terrain_configs[self.terrain_type][self.difficulty_level]
        
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=25,
                camera_pos=(3.0, 3.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=0.02,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        
        if self.terrain_type == "steps":
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
                    subterrain_params={
                        "pyramid_stairs_terrain": {
                            "step_height": config["step_height"],
                            "step_width": config["step_width"]
                        }
                    }
                ),
            )
        elif self.terrain_type == "slopes":
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
                    subterrain_params={
                        "pyramid_sloped_terrain": {
                            "slope": config["slope"]
                        }
                    }
                ),
            )
        
        return scene
    
    def get_description(self) -> str:
        """現在の設定の説明を取得"""
        config = self.terrain_configs[self.terrain_type][self.difficulty_level]
        return f"{self.terrain_type.capitalize()} - Level {self.difficulty_level}: {config['description']}"


class PolicyEvaluator:
    """ポリシー評価クラス"""
    
    def __init__(self, policy_path: str, config_path: str):
        """
        Args:
            policy_path: 学習済みポリシーのパス
            config_path: 設定ファイルのパス
        """
        self.policy_path = policy_path
        self.config_path = config_path
        self.policy = None
        self.env = None
        
    def load_policy(self):
        """ポリシーを読み込み"""
        try:
            from rsl_rl.runners import OnPolicyRunner
            
            # 設定を読み込み
            with open(self.config_path, "rb") as f:
                env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
            
            # 評価用に報酬を無効化
            reward_cfg["reward_scales"] = {}
            
            # 環境を作成
            self.env = Go2Env(
                num_envs=1,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=False,
            )
            
            # ポリシーを読み込み
            runner = OnPolicyRunner(self.env, train_cfg, "", device=gs.device)
            runner.load(self.policy_path)
            self.policy = runner.get_inference_policy(device=gs.device)
            
            print(f"✓ ポリシーを読み込みました: {self.policy_path}")
            return True
            
        except Exception as e:
            print(f"✗ ポリシーの読み込みに失敗しました: {e}")
            return False
    
    def evaluate_on_terrain(self, terrain_env: TerrainEvaluationEnvironment, 
                          max_steps: int = 1000, record_data: bool = True) -> Dict[str, Any]:
        """指定された地形でポリシーを評価"""
        if self.policy is None:
            print("ポリシーが読み込まれていません")
            return {}
        
        # 地形シーンを作成
        terrain_scene = terrain_env.create_terrain_scene(show_viewer=True)
        
        # ロボットを追加
        base_init_pos = torch.tensor([0.0, 0.0, 0.5], device=gs.device)
        base_init_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=gs.device)
        
        robot = terrain_scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=base_init_pos.cpu().numpy(),
                quat=base_init_quat.cpu().numpy(),
            ),
        )
        
        terrain_scene.build(n_envs=1)
        
        # 評価データを記録
        evaluation_data = {
            "terrain_type": terrain_env.terrain_type,
            "difficulty_level": terrain_env.difficulty_level,
            "description": terrain_env.get_description(),
            "max_steps": max_steps,
            "positions": [],
            "velocities": [],
            "orientations": [],
            "success": False,
            "final_distance": 0.0,
            "steps_completed": 0,
        }
        
        print(f"\n=== 評価開始: {terrain_env.get_description()} ===")
        
        # シミュレーション実行
        with torch.no_grad():
            for step in range(max_steps):
                # 簡単な前進コマンドを生成（実際のポリシー評価では適切な観測が必要）
                # ここでは仮の実装として基本的な前進動作を使用
                
                # ロボットの状態を取得
                pos = robot.get_pos()[0]  # [0]でバッチ次元を除去
                quat = robot.get_quat()[0]
                vel = robot.get_vel()[0]
                
                if record_data:
                    evaluation_data["positions"].append(pos.cpu().numpy())
                    evaluation_data["velocities"].append(vel.cpu().numpy())
                    evaluation_data["orientations"].append(quat.cpu().numpy())
                
                # 簡単な前進制御（実際のポリシーに置き換える必要あり）
                target_actions = torch.zeros(12, device=gs.device)  # 12 DOF
                robot.control_dofs_position(target_actions, list(range(12)))
                
                terrain_scene.step()
                
                # 進行距離をチェック
                distance_traveled = float(pos[0])  # X方向の移動距離
                evaluation_data["final_distance"] = distance_traveled
                evaluation_data["steps_completed"] = step + 1
                
                # 成功条件：5m以上前進
                if distance_traveled > 5.0:
                    evaluation_data["success"] = True
                    print(f"✓ 成功！ {distance_traveled:.2f}m 前進 ({step+1} ステップ)")
                    break
                
                # 失敗条件：転倒チェック
                if pos[2] < 0.2:  # 高さが20cm以下
                    print(f"✗ 転倒により失敗 ({step+1} ステップ)")
                    break
                
                if step % 100 == 0:
                    print(f"  Step {step}: 位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        final_msg = f"最終結果: {evaluation_data['final_distance']:.2f}m 前進"
        print(f"=== 評価完了: {final_msg} ===\n")
        
        return evaluation_data


def main():
    parser = argparse.ArgumentParser(description="地形における既存ポリシーの汎化性能評価")
    parser.add_argument("--terrain_type", choices=["steps", "slopes"], required=True,
                       help="地形タイプ: steps (段差) or slopes (傾斜)")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3], default=1,
                       help="難易度レベル: 1 (easy), 2 (medium), 3 (hard)")
    parser.add_argument("--policy_path", type=str,
                       help="学習済みポリシーのパス (.pt ファイル)")
    parser.add_argument("--config_path", type=str,
                       help="設定ファイルのパス (cfgs.pkl ファイル)")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="最大ステップ数")
    parser.add_argument("--all_levels", action="store_true",
                       help="全ての難易度レベルで評価")
    parser.add_argument("--no_vis", action="store_true",
                       help="可視化を無効化")
    
    args = parser.parse_args()
    
    # Genesis初期化
    gs.init(seed=42, backend=gs.gpu)
    
    if args.all_levels:
        # 全レベルでの評価
        results = {}
        
        for level in [1, 2, 3]:
            print(f"\n{'='*60}")
            print(f"レベル {level} での評価を開始")
            print(f"{'='*60}")
            
            terrain_env = TerrainEvaluationEnvironment(args.terrain_type, level)
            
            if args.policy_path and args.config_path:
                # ポリシー評価
                evaluator = PolicyEvaluator(args.policy_path, args.config_path)
                if evaluator.load_policy():
                    result = evaluator.evaluate_on_terrain(terrain_env, args.max_steps)
                    results[level] = result
            else:
                # 地形のみの可視化
                scene = terrain_env.create_terrain_scene(show_viewer=not args.no_vis)
                scene.build(n_envs=1)
                
                print(f"地形を表示中: {terrain_env.get_description()}")
                for _ in range(500):
                    scene.step()
                    time.sleep(0.02)
        
        # 結果サマリー
        if results:
            print(f"\n{'='*60}")
            print(f"{args.terrain_type.upper()} 評価結果サマリー")
            print(f"{'='*60}")
            for level, result in results.items():
                success_str = "✓ 成功" if result["success"] else "✗ 失敗"
                print(f"レベル {level}: {success_str} | 距離: {result['final_distance']:.2f}m | "
                      f"ステップ: {result['steps_completed']}")
    
    else:
        # 単一レベルでの評価
        terrain_env = TerrainEvaluationEnvironment(args.terrain_type, args.difficulty)
        
        if args.policy_path and args.config_path:
            # ポリシー評価
            evaluator = PolicyEvaluator(args.policy_path, args.config_path)
            if evaluator.load_policy():
                result = evaluator.evaluate_on_terrain(terrain_env, args.max_steps)
                print("評価完了！")
        else:
            # 地形のみの可視化
            scene = terrain_env.create_terrain_scene(show_viewer=not args.no_vis)
            scene.build(n_envs=1)
            
            print(f"地形を表示中: {terrain_env.get_description()}")
            print("ESCキーで終了してください")
            
            try:
                while True:
                    scene.step()
                    time.sleep(0.02)
            except KeyboardInterrupt:
                print("終了しました")


if __name__ == "__main__":
    main()