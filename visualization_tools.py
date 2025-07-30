"""
地形評価結果の可視化・記録ツール
軌跡、パフォーマンスメトリクス、比較グラフの生成機能
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
import imageio

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


class TerrainVisualizationTools:
    """地形評価結果の可視化ツール"""
    
    def __init__(self, results_dir: str = "terrain_benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # カラーパレット
        self.colors = {
            'success': '#2E8B57',
            'failure': '#DC143C',
            'steps': '#4169E1',
            'slopes': '#FF8C00',
            'level1': '#90EE90',
            'level2': '#FFD700',
            'level3': '#FF6347'
        }
    
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """結果ファイルを読み込み"""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def plot_success_rate_heatmap(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """成功率のヒートマップを生成"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        policies = list(results["summary"]["success_rates"].keys())
        terrain_types = results["configuration"]["terrain_types"]
        difficulty_levels = results["configuration"]["difficulty_levels"]
        
        for i, terrain_type in enumerate(terrain_types):
            data = []
            for policy in policies:
                row = []
                for level in difficulty_levels:
                    rate = results["summary"]["success_rates"][policy].get(terrain_type, {}).get(str(level), 0)
                    row.append(rate * 100)
                data.append(row)
            
            df = pd.DataFrame(data, 
                            index=policies, 
                            columns=[f"Level {level}" for level in difficulty_levels])
            
            sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', 
                       ax=axes[i], cbar_kws={'label': '成功率 (%)'})
            axes[i].set_title(f'{terrain_type.capitalize()} - 成功率ヒートマップ')
            axes[i].set_ylabel('ポリシー')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / "success_rate_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_performance_comparison(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """パフォーマンス比較グラフを生成"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        policies = list(results["summary"]["success_rates"].keys())
        terrain_types = results["configuration"]["terrain_types"]
        difficulty_levels = results["configuration"]["difficulty_levels"]
        
        metrics = [
            ("success_rates", "成功率 (%)", "success_rates"),
            ("average_distances", "平均移動距離 (m)", "average_distances"),
            ("stability_scores", "安定性スコア", "stability_scores"),
            ("completion_times", "完了時間 (s)", "completion_times")
        ]
        
        for idx, (metric_key, ylabel, summary_key) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            x = np.arange(len(difficulty_levels))
            width = 0.35
            
            for i, terrain_type in enumerate(terrain_types):
                offset = (i - 0.5) * width
                
                for j, policy in enumerate(policies):
                    values = []
                    for level in difficulty_levels:
                        value = results["summary"][summary_key][policy].get(terrain_type, {}).get(str(level), 0)
                        if metric_key == "success_rates":
                            value *= 100
                        values.append(value)
                    
                    color = self.colors.get(terrain_type, f'C{i}')
                    alpha = 0.7 + j * 0.3 / len(policies)
                    
                    bars = ax.bar(x + offset + j * width / len(policies), values, 
                                width / len(policies), label=f'{policy} ({terrain_type})',
                                color=color, alpha=alpha)
            
            ax.set_xlabel('難易度レベル')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel}の比較')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Level {level}' for level in difficulty_levels])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / "performance_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_trajectory_analysis(self, results: Dict[str, Any], 
                                policy_name: str, terrain_type: str, difficulty: int,
                                save_path: Optional[str] = None) -> str:
        """軌跡分析グラフを生成"""
        
        runs = results["policies"][policy_name]["results"][terrain_type][str(difficulty)]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for run_idx, run_data in enumerate(runs):
            if not run_data["trajectory"]["positions"]:
                continue
                
            positions = np.array(run_data["trajectory"]["positions"])
            velocities = np.array(run_data["trajectory"]["velocities"])
            timestamps = np.array(run_data["trajectory"]["timestamps"])
            
            # 2D軌跡プロット
            ax = axes[0, 0]
            color = self.colors['success'] if run_data["success"] else self.colors['failure']
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=color, alpha=0.7, label=f'Run {run_idx+1}')
            ax.set_xlabel('X位置 (m)')
            ax.set_ylabel('Y位置 (m)')
            ax.set_title('2D軌跡')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            # 高度変化
            ax = axes[0, 1]
            ax.plot(timestamps, positions[:, 2], color=color, alpha=0.7)
            ax.set_xlabel('時間 (s)')
            ax.set_ylabel('高度 (m)')
            ax.set_title('高度変化')
            ax.grid(True, alpha=0.3)
            
            # 速度プロファイル
            ax = axes[1, 0]
            speeds = np.linalg.norm(velocities, axis=1)
            ax.plot(timestamps, speeds, color=color, alpha=0.7)
            ax.set_xlabel('時間 (s)')
            ax.set_ylabel('速度 (m/s)')
            ax.set_title('速度プロファイル')
            ax.grid(True, alpha=0.3)
            
            # 進行距離
            ax = axes[1, 1]
            ax.plot(timestamps, positions[:, 0], color=color, alpha=0.7)
            ax.set_xlabel('時間 (s)')
            ax.set_ylabel('X進行距離 (m)')
            ax.set_title('進行距離')
            ax.grid(True, alpha=0.3)
        
        axes[0, 0].legend()
        plt.suptitle(f'{policy_name} - {terrain_type} Level {difficulty} 軌跡分析')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / f"trajectory_{policy_name}_{terrain_type}_L{difficulty}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_animated_trajectory(self, results: Dict[str, Any],
                                 policy_name: str, terrain_type: str, difficulty: int,
                                 run_index: int = 0, save_path: Optional[str] = None) -> str:
        """軌跡のアニメーションを作成"""
        
        run_data = results["policies"][policy_name]["results"][terrain_type][str(difficulty)][run_index]
        positions = np.array(run_data["trajectory"]["positions"])
        timestamps = np.array(run_data["trajectory"]["timestamps"])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 地形の概要を描画（簡易版）
        if terrain_type == "steps":
            # 段差を矩形で表現
            step_heights = [0.05, 0.10, 0.15]
            step_height = step_heights[difficulty - 1]
            
            for i in range(5):
                rect = patches.Rectangle((i*2, 0), 2, step_height * i, 
                                       linewidth=1, edgecolor='gray', 
                                       facecolor='lightgray', alpha=0.5)
                ax.add_patch(rect)
        
        # アニメーション用のプロット要素
        line, = ax.plot([], [], 'b-', linewidth=2, label='軌跡')
        point, = ax.plot([], [], 'ro', markersize=8, label='現在位置')
        
        ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
        ax.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
        ax.set_xlabel('X位置 (m)')
        ax.set_ylabel('Y位置 (m)')
        ax.set_title(f'{policy_name} - {terrain_type} Level {difficulty} 軌跡アニメーション')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        def animate(frame):
            # 現在のフレームまでの軌跡を描画
            line.set_data(positions[:frame, 0], positions[:frame, 1])
            point.set_data([positions[frame, 0]], [positions[frame, 1]])
            
            # タイムスタンプを表示
            time_text = f'時間: {timestamps[frame]:.2f}s'
            ax.set_title(f'{policy_name} - {terrain_type} Level {difficulty} 軌跡アニメーション ({time_text})')
            
            return line, point
        
        # アニメーション作成
        anim = FuncAnimation(fig, animate, frames=len(positions), 
                           interval=50, blit=False, repeat=True)
        
        if save_path is None:
            save_path = self.results_dir / f"trajectory_animation_{policy_name}_{terrain_type}_L{difficulty}.gif"
        
        anim.save(save_path, writer='pillow', fps=20)
        plt.close()
        
        return str(save_path)
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """包括的な可視化レポートを生成"""
        
        print("可視化レポートを生成中...")
        
        # 各種グラフを生成
        heatmap_path = self.plot_success_rate_heatmap(results)
        comparison_path = self.plot_performance_comparison(results)
        
        # 各ポリシーの軌跡分析
        trajectory_paths = []
        for policy_name in results["policies"].keys():
            for terrain_type in results["configuration"]["terrain_types"]:
                for difficulty in results["configuration"]["difficulty_levels"]:
                    if str(difficulty) in results["policies"][policy_name]["results"][terrain_type]:
                        path = self.plot_trajectory_analysis(results, policy_name, terrain_type, difficulty)
                        trajectory_paths.append(path)
        
        # HTMLレポート生成
        html_content = self._generate_html_report(results, heatmap_path, comparison_path, trajectory_paths)
        
        report_path = self.results_dir / "visualization_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"可視化レポートを生成しました: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self, results: Dict[str, Any], 
                            heatmap_path: str, comparison_path: str, 
                            trajectory_paths: List[str]) -> str:
        """HTMLレポートを生成"""
        
        html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>地形汎化性能評価レポート</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                .image-container img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>地形汎化性能評価レポート</h1>
            
            <div class="summary">
                <h2>実行概要</h2>
                <p><strong>実行日時:</strong> {results['timestamp']}</p>
                <p><strong>評価ポリシー数:</strong> {len(results['policies'])}</p>
                <p><strong>地形タイプ:</strong> {', '.join(results['configuration']['terrain_types'])}</p>
                <p><strong>難易度レベル:</strong> {', '.join(map(str, results['configuration']['difficulty_levels']))}</p>
            </div>
            
            <h2>成功率ヒートマップ</h2>
            <div class="image-container">
                <img src="{os.path.basename(heatmap_path)}" alt="成功率ヒートマップ">
            </div>
            
            <h2>パフォーマンス比較</h2>
            <div class="image-container">
                <img src="{os.path.basename(comparison_path)}" alt="パフォーマンス比較">
            </div>
            
            <h2>軌跡分析</h2>
        """
        
        for path in trajectory_paths:
            html += f"""
            <div class="image-container">
                <img src="{os.path.basename(path)}" alt="軌跡分析">
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """メイン関数 - コマンドライン使用例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="地形評価結果の可視化")
    parser.add_argument("--results_file", required=True,
                       help="結果JSONファイルのパス")
    parser.add_argument("--output_dir", default="terrain_benchmark_results",
                       help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    # 可視化ツールを初期化
    viz_tools = TerrainVisualizationTools(args.output_dir)
    
    # 結果を読み込み
    results = viz_tools.load_results(args.results_file)
    
    # 包括的なレポートを生成
    report_path = viz_tools.generate_comprehensive_report(results)
    
    print(f"可視化完了！レポート: {report_path}")


if __name__ == "__main__":
    main()