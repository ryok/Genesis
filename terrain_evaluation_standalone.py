"""
地形汎化性能評価ツール（スタンドアロン版）
Genesis依存を最小限にした簡易評価スクリプト
"""

import argparse
import numpy as np
import time

# 簡易的な地形パラメータ出力
def print_terrain_config(terrain_type: str, difficulty: int):
    """地形設定を表示"""
    
    # 段差設定
    step_configs = {
        1: {"height": 0.05, "width": 1.0, "desc": "低い段差 (5cm)"},
        2: {"height": 0.10, "width": 0.8, "desc": "中程度の段差 (10cm)"},  
        3: {"height": 0.15, "width": 0.6, "desc": "高い段差 (15cm)"},
    }
    
    # 傾斜設定
    slope_configs = {
        1: {"angle": 6, "slope": 0.1, "desc": "緩やかな傾斜 (6度)"},
        2: {"angle": 11, "slope": 0.2, "desc": "中程度の傾斜 (11度)"},
        3: {"angle": 17, "slope": 0.3, "desc": "急な傾斜 (17度)"},
    }
    
    print(f"\n=== 地形設定 ===")
    print(f"タイプ: {terrain_type}")
    print(f"難易度: レベル {difficulty}")
    
    if terrain_type == "steps":
        config = step_configs[difficulty]
        print(f"説明: {config['desc']}")
        print(f"段差高さ: {config['height']}m")
        print(f"段差幅: {config['width']}m")
        print(f"\n地形レイアウト (3x3 グリッド):")
        print("  [平地] [段差] [平地]")
        print("  [段差] [段差] [段差]")
        print("  [平地] [段差] [平地]")
        
    elif terrain_type == "slopes":
        config = slope_configs[difficulty]
        print(f"説明: {config['desc']}")
        print(f"傾斜角度: {config['angle']}度")
        print(f"slope値: {config['slope']}")
        print(f"\n地形レイアウト (3x3 グリッド):")
        print("  [平地] [傾斜] [平地]")
        print("  [傾斜] [傾斜] [傾斜]")
        print("  [平地] [傾斜] [平地]")
    
    print("\n地形パラメータ (Genesis用):")
    print(f"  subterrain_size: (4.0, 4.0)")
    print(f"  horizontal_scale: 0.25")
    print(f"  vertical_scale: 0.01")
    

def generate_example_code(terrain_type: str, difficulty: int):
    """Genesis用のサンプルコードを生成"""
    
    if terrain_type == "steps":
        step_heights = [0.05, 0.10, 0.15]
        step_widths = [1.0, 0.8, 0.6]
        
        code = f'''
# Genesis での地形生成コード例:

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
        subterrain_params={{
            "pyramid_stairs_terrain": {{
                "step_height": {step_heights[difficulty-1]},
                "step_width": {step_widths[difficulty-1]}
            }}
        }}
    ),
)
'''
    else:  # slopes
        slopes = [0.1, 0.2, 0.3]
        
        code = f'''
# Genesis での地形生成コード例:

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
        subterrain_params={{
            "pyramid_sloped_terrain": {{
                "slope": {slopes[difficulty-1]}
            }}
        }}
    ),
)
'''
    
    print(code)


def analyze_terrain_challenge(terrain_type: str, difficulty: int):
    """地形の難易度を分析"""
    
    print(f"\n=== 難易度分析 ===")
    
    if terrain_type == "steps":
        challenges = {
            1: [
                "低い段差は基本的な歩行制御で対応可能",
                "足の持ち上げ高さの調整が必要",
                "重心移動は比較的小さい"
            ],
            2: [
                "中程度の段差では動的バランス制御が重要",
                "段差幅が狭くなり、正確な足配置が必要",
                "前進時の重心制御がより重要に"
            ],
            3: [
                "高い段差は高度な全身協調が必要",
                "狭い段差幅により転倒リスクが増大",
                "登り降りで異なる制御戦略が必要"
            ]
        }
    else:  # slopes
        challenges = {
            1: [
                "緩やかな傾斜は基本的な姿勢制御で対応可能",
                "わずかな重心調整で安定歩行を維持",
                "滑りリスクは低い"
            ],
            2: [
                "中程度の傾斜では前後の重心バランスが重要",
                "上り下りで異なる歩行パターンが必要",
                "足の接地角度の調整が必要"
            ],
            3: [
                "急傾斜は高度な姿勢制御と踏ん張りが必要",
                "滑落リスクが高く、グリップ力が重要",
                "エネルギー効率的な歩行が困難"
            ]
        }
    
    print(f"レベル {difficulty} の主な課題:")
    for i, challenge in enumerate(challenges[difficulty], 1):
        print(f"  {i}. {challenge}")
    

def suggest_evaluation_metrics(terrain_type: str):
    """評価メトリクスの提案"""
    
    print(f"\n=== 推奨評価メトリクス ===")
    
    common_metrics = [
        "成功率: 目標距離（5-8m）の到達率",
        "移動距離: 転倒/失敗までの前進距離",
        "完了時間: タスク完了までの時間",
        "安定性スコア: 姿勢の安定性指標"
    ]
    
    if terrain_type == "steps":
        specific_metrics = [
            "段差成功率: 各段差の登り降り成功率",
            "足配置精度: 段差上での足の配置誤差",
            "重心軌跡: 段差昇降時の重心移動パターン",
            "回復能力: バランス崩れからの復帰頻度"
        ]
    else:  # slopes
        specific_metrics = [
            "傾斜角度耐性: 安定歩行可能な最大傾斜",
            "滑り頻度: 足の滑り発生回数",
            "エネルギー効率: 傾斜歩行時の消費エネルギー",
            "速度維持率: 平地比での歩行速度"
        ]
    
    print("共通メトリクス:")
    for metric in common_metrics:
        print(f"  • {metric}")
    
    print(f"\n{terrain_type}固有メトリクス:")
    for metric in specific_metrics:
        print(f"  • {metric}")


def main():
    parser = argparse.ArgumentParser(
        description="地形汎化性能評価ツール（スタンドアロン版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python terrain_evaluation_standalone.py --terrain steps --difficulty 2
  python terrain_evaluation_standalone.py --terrain slopes --all_levels
  python terrain_evaluation_standalone.py --terrain steps --difficulty 3 --show_code
        """
    )
    
    parser.add_argument("--terrain", choices=["steps", "slopes"], required=True,
                       help="地形タイプ")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3],
                       help="難易度レベル (1: 易, 2: 中, 3: 難)")
    parser.add_argument("--all_levels", action="store_true",
                       help="全レベルの情報を表示")
    parser.add_argument("--show_code", action="store_true",
                       help="Genesis用のサンプルコードを表示")
    
    args = parser.parse_args()
    
    print("="*60)
    print("地形汎化性能評価ツール - 設定情報")
    print("="*60)
    
    if args.all_levels:
        for level in [1, 2, 3]:
            print(f"\n{'='*20} レベル {level} {'='*20}")
            print_terrain_config(args.terrain, level)
            analyze_terrain_challenge(args.terrain, level)
            
            if args.show_code:
                generate_example_code(args.terrain, level)
    else:
        if args.difficulty is None:
            parser.error("--difficulty または --all_levels を指定してください")
        
        print_terrain_config(args.terrain, args.difficulty)
        analyze_terrain_challenge(args.terrain, args.difficulty)
        
        if args.show_code:
            generate_example_code(args.terrain, args.difficulty)
    
    suggest_evaluation_metrics(args.terrain)
    
    print("\n" + "="*60)
    print("注意: このスタンドアロン版は設定情報の表示のみです。")
    print("実際のシミュレーションには Genesis の正常なインストールが必要です。")
    print("="*60)


if __name__ == "__main__":
    main()