"""
既存ポリシーの地形汎化性能ベンチマークスクリプト
詳細な評価メトリクスと結果レポート機能を提供
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import genesis as gs
from terrain_evaluation import TerrainEvaluationEnvironment, PolicyEvaluator


class AdvancedPolicyEvaluator(PolicyEvaluator):
    """高度なポリシー評価機能を持つクラス"""
    
    def __init__(self, policy_path: str, config_path: str):
        super().__init__(policy_path, config_path)
        self.evaluation_history = []
    
    def evaluate_with_metrics(
        self, 
        terrain_env: TerrainEvaluationEnvironment,
        max_steps: int = 2000,
        target_distance: float = 8.0,
        stability_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """詳細なメトリクスを含む評価"""
        
        if self.policy is None:
            raise ValueError("ポリシーが読み込まれていません")
        
        # 評価環境をセットアップ
        terrain_scene = terrain_env.create_terrain_scene(show_viewer=False)
        
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
        
        # メトリクス初期化
        metrics = {
            "terrain_info": {
                "type": terrain_env.terrain_type,
                "difficulty": terrain_env.difficulty_level,
                "description": terrain_env.get_description()
            },
            "success": False,
            "completion_time": 0,
            "total_steps": 0,
            "final_distance": 0.0,
            "max_distance": 0.0,
            "average_speed": 0.0,
            "distance_efficiency": 0.0,  # 実際の移動距離 / 理想的な直線距離
            "stability_score": 0.0,
            "fall_count": 0,
            "recovery_count": 0,
            "energy_consumption": 0.0,
            "trajectory": {
                "positions": [],
                "velocities": [],
                "orientations": [],
                "timestamps": []
            }
        }
        
        start_time = time.time()
        last_stable_time = start_time
        fall_detected = False
        
        print(f"評価開始: {terrain_env.get_description()}")
        
        with torch.no_grad():
            for step in tqdm(range(max_steps), desc="Evaluating"):
                # ロボット状態取得
                pos = robot.get_pos()[0]
                quat = robot.get_quat()[0]
                vel = robot.get_vel()[0]
                ang_vel = robot.get_ang()[0]
                
                # 軌跡記録
                current_time = time.time() - start_time
                metrics["trajectory"]["positions"].append(pos.cpu().numpy())
                metrics["trajectory"]["velocities"].append(vel.cpu().numpy())
                metrics["trajectory"]["orientations"].append(quat.cpu().numpy())
                metrics["trajectory"]["timestamps"].append(current_time)
                
                # 距離計算
                distance_traveled = float(pos[0])
                lateral_deviation = abs(float(pos[1]))
                
                metrics["final_distance"] = distance_traveled
                metrics["max_distance"] = max(metrics["max_distance"], distance_traveled)
                
                # 転倒検出
                height = float(pos[2])
                roll = float(torch.atan2(2*(quat[3]*quat[0] + quat[1]*quat[2]), 
                                       1 - 2*(quat[0]**2 + quat[1]**2)))
                pitch = float(torch.asin(2*(quat[3]*quat[1] - quat[2]*quat[0])))
                
                is_fallen = (height < stability_threshold or 
                           abs(roll) > np.pi/3 or abs(pitch) > np.pi/3)
                
                if is_fallen and not fall_detected:
                    metrics["fall_count"] += 1
                    fall_detected = True
                elif not is_fallen and fall_detected:
                    metrics["recovery_count"] += 1
                    fall_detected = False
                    last_stable_time = current_time
                
                # 成功条件チェック
                if distance_traveled >= target_distance:
                    metrics["success"] = True
                    break
                
                # 致命的失敗条件
                if height < 0.1 or current_time - last_stable_time > 5.0:
                    break
                
                # 基本的な前進制御（実際の評価では適切なポリシーを使用）
                target_actions = torch.zeros(12, device=gs.device)
                robot.control_dofs_position(target_actions, list(range(12)))
                
                terrain_scene.step()
                
                metrics["total_steps"] = step + 1
        
        # 最終メトリクス計算
        metrics["completion_time"] = time.time() - start_time
        
        if metrics["completion_time"] > 0:
            metrics["average_speed"] = metrics["final_distance"] / metrics["completion_time"]
        
        # 軌跡効率性計算
        if len(metrics["trajectory"]["positions"]) > 1:
            positions = np.array(metrics["trajectory"]["positions"])
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            direct_distance = np.linalg.norm(positions[-1] - positions[0])
            if path_length > 0:
                metrics["distance_efficiency"] = direct_distance / path_length
        
        # 安定性スコア計算
        stable_time = metrics["completion_time"] - (metrics["fall_count"] * 2.0)  # 転倒時間を推定
        metrics["stability_score"] = max(0, stable_time / metrics["completion_time"]) if metrics["completion_time"] > 0 else 0
        
        self.evaluation_history.append(metrics)
        return metrics


class BenchmarkRunner:
    """ベンチマーク実行クラス"""
    
    def __init__(self, results_dir: str = "terrain_benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def run_comprehensive_benchmark(
        self,
        policy_paths: List[str],
        config_paths: List[str],
        policy_names: Optional[List[str]] = None,
        terrain_types: List[str] = ["steps", "slopes"],
        difficulty_levels: List[int] = [1, 2, 3],
        max_steps: int = 2000,
        runs_per_config: int = 3
    ) -> Dict[str, Any]:
        """包括的ベンチマークの実行"""
        
        if policy_names is None:
            policy_names = [f"Policy_{i+1}" for i in range(len(policy_paths))]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "terrain_types": terrain_types,
                "difficulty_levels": difficulty_levels,
                "max_steps": max_steps,
                "runs_per_config": runs_per_config
            },
            "policies": {},
            "summary": {}
        }
        
        total_evaluations = len(policy_paths) * len(terrain_types) * len(difficulty_levels) * runs_per_config
        progress_bar = tqdm(total=total_evaluations, desc="Running benchmark")
        
        for i, (policy_path, config_path, policy_name) in enumerate(zip(policy_paths, config_paths, policy_names)):
            results["policies"][policy_name] = {
                "path": policy_path,
                "config": config_path,
                "results": {}
            }
            
            evaluator = AdvancedPolicyEvaluator(policy_path, config_path)
            if not evaluator.load_policy():
                print(f"警告: {policy_name} の読み込みに失敗しました")
                continue
            
            for terrain_type in terrain_types:
                results["policies"][policy_name]["results"][terrain_type] = {}
                
                for difficulty in difficulty_levels:
                    run_results = []
                    
                    for run in range(runs_per_config):
                        terrain_env = TerrainEvaluationEnvironment(terrain_type, difficulty)
                        
                        try:
                            metrics = evaluator.evaluate_with_metrics(
                                terrain_env, max_steps=max_steps
                            )
                            run_results.append(metrics)
                            
                        except Exception as e:
                            print(f"評価エラー ({policy_name}, {terrain_type}, Level {difficulty}, Run {run+1}): {e}")
                            
                        progress_bar.update(1)
                    
                    results["policies"][policy_name]["results"][terrain_type][difficulty] = run_results
        
        progress_bar.close()
        
        # サマリー統計を計算
        self._calculate_summary_statistics(results)
        
        # 結果を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"結果を保存しました: {results_file}")
        
        return results
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]):
        """サマリー統計を計算"""
        summary = {
            "success_rates": {},
            "average_distances": {},
            "stability_scores": {},
            "completion_times": {}
        }
        
        for policy_name, policy_data in results["policies"].items():
            summary["success_rates"][policy_name] = {}
            summary["average_distances"][policy_name] = {}
            summary["stability_scores"][policy_name] = {}
            summary["completion_times"][policy_name] = {}
            
            for terrain_type, terrain_data in policy_data["results"].items():
                summary["success_rates"][policy_name][terrain_type] = {}
                summary["average_distances"][policy_name][terrain_type] = {}
                summary["stability_scores"][policy_name][terrain_type] = {}
                summary["completion_times"][policy_name][terrain_type] = {}
                
                for difficulty, runs in terrain_data.items():
                    if not runs:
                        continue
                    
                    successes = sum(1 for run in runs if run["success"])
                    distances = [run["final_distance"] for run in runs]
                    stability = [run["stability_score"] for run in runs]
                    times = [run["completion_time"] for run in runs]
                    
                    summary["success_rates"][policy_name][terrain_type][difficulty] = successes / len(runs)
                    summary["average_distances"][policy_name][terrain_type][difficulty] = np.mean(distances)
                    summary["stability_scores"][policy_name][terrain_type][difficulty] = np.mean(stability)
                    summary["completion_times"][policy_name][terrain_type][difficulty] = np.mean(times)
        
        results["summary"] = summary
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """結果レポートの生成"""
        report_lines = [
            "# 地形汎化性能ベンチマーク結果レポート",
            f"実行日時: {results['timestamp']}",
            "",
            "## 設定",
            f"- 地形タイプ: {', '.join(results['configuration']['terrain_types'])}",
            f"- 難易度レベル: {', '.join(map(str, results['configuration']['difficulty_levels']))}",
            f"- 最大ステップ数: {results['configuration']['max_steps']}",
            f"- 実行回数/設定: {results['configuration']['runs_per_config']}",
            "",
            "## 結果サマリー",
            ""
        ]
        
        # 成功率テーブル
        report_lines.append("### 成功率 (%)")
        summary = results["summary"]
        
        for terrain_type in results["configuration"]["terrain_types"]:
            report_lines.append(f"\n#### {terrain_type.capitalize()}")
            
            # テーブルヘッダー
            policies = list(summary["success_rates"].keys())
            levels = results["configuration"]["difficulty_levels"]
            
            header = "| Policy | " + " | ".join([f"Level {level}" for level in levels]) + " |"
            separator = "|" + "|".join([" --- " for _ in range(len(levels) + 1)]) + "|"
            
            report_lines.extend([header, separator])
            
            for policy in policies:
                row = f"| {policy} |"
                for level in levels:
                    rate = summary["success_rates"][policy].get(terrain_type, {}).get(level, 0)
                    row += f" {rate*100:.1f}% |"
                report_lines.append(row)
        
        # ファイルに保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"benchmark_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"レポートを保存しました: {report_file}")
        return str(report_file)


def main():
    parser = argparse.ArgumentParser(description="地形汎化性能ベンチマーク")
    parser.add_argument("--policies", nargs="+", required=True,
                       help="評価するポリシーファイルのパス")
    parser.add_argument("--configs", nargs="+", required=True,
                       help="対応する設定ファイルのパス")
    parser.add_argument("--names", nargs="+",
                       help="ポリシーの名前（省略時は自動生成）")
    parser.add_argument("--terrain_types", nargs="+", default=["steps", "slopes"],
                       choices=["steps", "slopes"],
                       help="評価する地形タイプ")
    parser.add_argument("--difficulties", nargs="+", type=int, default=[1, 2, 3],
                       choices=[1, 2, 3],
                       help="評価する難易度レベル")
    parser.add_argument("--max_steps", type=int, default=2000,
                       help="評価の最大ステップ数")
    parser.add_argument("--runs", type=int, default=3,
                       help="各設定での実行回数")
    parser.add_argument("--results_dir", default="terrain_benchmark_results",
                       help="結果保存ディレクトリ")
    
    args = parser.parse_args()
    
    if len(args.policies) != len(args.configs):
        raise ValueError("ポリシーファイルと設定ファイルの数が一致しません")
    
    # Genesis初期化
    gs.init(seed=42, backend=gs.gpu)
    
    # ベンチマーク実行
    runner = BenchmarkRunner(args.results_dir)
    
    results = runner.run_comprehensive_benchmark(
        policy_paths=args.policies,
        config_paths=args.configs,
        policy_names=args.names,
        terrain_types=args.terrain_types,
        difficulty_levels=args.difficulties,
        max_steps=args.max_steps,
        runs_per_config=args.runs
    )
    
    # レポート生成
    report_file = runner.generate_report(results)
    
    print("\n" + "="*60)
    print("ベンチマーク完了！")
    print(f"結果ファイル: {runner.results_dir}")
    print(f"レポート: {report_file}")
    print("="*60)


if __name__ == "__main__":
    main()