#!/usr/bin/env python3
"""
基本的な地形評価の使用例
terrain_evaluation_tools の基本的な使い方を示すサンプルスクリプト
"""

import os
import sys
import subprocess
from pathlib import Path

# terrain_evaluation_tools のルートディレクトリを取得
TOOLS_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = TOOLS_ROOT / "scripts"
OUTPUTS_DIR = TOOLS_ROOT / "outputs"

def run_basic_evaluation():
    """基本的な地形評価を実行"""
    print("=== 基本的な地形評価の実行例 ===\n")
    
    # 1. シンプルな評価（ポリシーなし）
    print("1. シンプルな段差評価（レベル2）")
    cmd = [
        "python", str(SCRIPTS_DIR / "terrain_evaluation_final.py"),
        "--terrain", "steps",
        "--difficulty", "2"
    ]
    print(f"実行コマンド: {' '.join(cmd)}")
    print("説明: 学習済みポリシーなしで10cm段差を評価")
    print()
    
    # 2. 動画記録付き評価
    print("2. 動画記録付き評価")
    video_dir = OUTPUTS_DIR / "videos" / "examples"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", str(SCRIPTS_DIR / "terrain_evaluation_video_recorder_fixed.py"),
        "--terrain", "slopes",
        "--difficulty", "1", 
        "--save_video",
        "--video_dir", str(video_dir)
    ]
    print(f"実行コマンド: {' '.join(cmd)}")
    print("説明: 6度傾斜での動作を動画記録")
    print()
    
    # 3. 地形可視化確認
    print("3. 地形可視化確認")
    cmd = [
        "python", str(SCRIPTS_DIR / "terrain_evaluation_visualization_enhanced.py"),
        "--terrain", "steps",
        "--difficulty", "3",
        "--save_video"
    ]
    print(f"実行コマンド: {' '.join(cmd)}")
    print("説明: 15cm段差地形の可視化と動画記録")
    print()
    
    # 4. 全レベル比較
    print("4. 全レベル比較")
    print("段差地形の全レベル評価:")
    for level in [1, 2, 3]:
        cmd = [
            "python", str(SCRIPTS_DIR / "terrain_evaluation_final.py"),
            "--terrain", "steps",
            "--difficulty", str(level)
        ]
        print(f"  レベル{level}: {' '.join(cmd)}")
    print()

def run_batch_evaluation():
    """バッチ評価の例"""
    print("=== バッチ評価スクリプトの例 ===\n")
    
    batch_script = """#!/bin/bash
# 全地形・全レベルでの包括的評価

SCRIPTS_DIR="./scripts"
OUTPUT_DIR="./outputs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "地形汎化性能評価 - バッチ実行開始: $TIMESTAMP"

# 段差地形評価
echo "=== 段差地形評価 ==="
for level in 1 2 3; do
    echo "段差レベル $level を評価中..."
    python "$SCRIPTS_DIR/terrain_evaluation_final.py" \\
        --terrain steps --difficulty $level \\
        > "$OUTPUT_DIR/data/steps_level${level}_$TIMESTAMP.log" 2>&1
done

# 傾斜地形評価  
echo "=== 傾斜地形評価 ==="
for level in 1 2 3; do
    echo "傾斜レベル $level を評価中..."
    python "$SCRIPTS_DIR/terrain_evaluation_final.py" \\
        --terrain slopes --difficulty $level \\
        > "$OUTPUT_DIR/data/slopes_level${level}_$TIMESTAMP.log" 2>&1
done

# 動画記録付き失敗例記録
echo "=== 失敗例動画記録 ==="
python "$SCRIPTS_DIR/terrain_evaluation_video_recorder_fixed.py" \\
    --terrain steps --difficulty 2 \\
    --save_video --video_dir "$OUTPUT_DIR/videos/failure_cases"

echo "バッチ評価完了: $TIMESTAMP"
"""
    
    print(batch_script)

def run_policy_evaluation_example():
    """学習済みポリシーでの評価例"""
    print("=== 学習済みポリシー使用例 ===\n")
    
    print("学習済みポリシーがある場合の評価:")
    
    # 実際のポリシーパスの例
    policy_examples = [
        "logs/go2-walking/model_100.pt",
        "checkpoints/go2_policy_final.pt", 
        "trained_models/quadruped_locomotion.pth"
    ]
    
    for i, policy_path in enumerate(policy_examples, 1):
        print(f"{i}. ポリシー: {policy_path}")
        cmd = [
            "python", str(SCRIPTS_DIR / "terrain_evaluation_final.py"),
            "--terrain", "steps",
            "--difficulty", "2",
            "--policy", policy_path
        ]
        print(f"   コマンド: {' '.join(cmd)}")
        print()
    
    print("注意: ポリシーファイルが存在しない場合は、シンプルな歩行パターンが使用されます。")

def show_directory_structure():
    """整理されたディレクトリ構造を表示"""
    print("=== 整理されたディレクトリ構造 ===\n")
    
    structure = """
terrain_evaluation_tools/
├── README.md                   # メインドキュメント
├── scripts/                    # 実行スクリプト
│   ├── terrain_evaluation_final.py                    # 基本評価
│   ├── terrain_evaluation_video_recorder_fixed.py     # 動画記録
│   ├── terrain_evaluation_visualization_enhanced.py   # 可視化
│   └── terrain_evaluation_video_debug.py              # デバッグ
├── configs/                    # 設定ファイル
│   └── default_config.yaml     # デフォルト設定
├── docs/                       # ドキュメント
│   └── TERRAIN_TOOLS_README.md # 詳細マニュアル
├── outputs/                    # 出力ファイル
│   ├── videos/                 # 動画記録
│   ├── reports/                # 実験レポート
│   └── data/                   # 実験データ
└── examples/                   # 使用例
    └── basic_evaluation.py     # このファイル
"""
    print(structure)

def main():
    """メイン関数"""
    print("地形汎化性能評価ツール - 使用例")
    print("=" * 50)
    
    # ディレクトリ構造表示
    show_directory_structure()
    
    # 基本評価例
    run_basic_evaluation()
    
    # バッチ評価例
    run_batch_evaluation()
    
    # ポリシー評価例
    run_policy_evaluation_example()
    
    print("\n" + "=" * 50)
    print("詳細な使用方法は README.md を参照してください。")
    print("問題が発生した場合は docs/TERRAIN_TOOLS_README.md を確認してください。")

if __name__ == "__main__":
    main()