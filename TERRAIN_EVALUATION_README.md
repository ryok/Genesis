# 地形汎化性能評価ツール

複雑な地形において既存の歩行ポリシーがどこまで汎化するかを観察・記録するためのツールセットです。段差と傾斜の強さを3段階で比較できる環境を提供します。

## 機能概要

- **段差地形**: 3段階（5cm, 10cm, 15cm）の段差を持つ地形
- **傾斜地形**: 3段階（6度, 11度, 17度）の傾斜を持つ地形
- **詳細評価**: 成功率、移動距離、安定性、軌跡分析などの包括的メトリクス
- **可視化**: ヒートマップ、比較グラフ、軌跡アニメーションの自動生成
- **レポート**: HTML形式の包括的評価レポート

## ファイル構成

```
├── terrain_evaluation.py          # 基本的な地形評価スクリプト
├── policy_terrain_benchmark.py    # 包括的ベンチマークツール
├── visualization_tools.py         # 可視化・レポート生成ツール
└── TERRAIN_EVALUATION_README.md   # このファイル
```

## 必要な依存関係

```bash
pip install matplotlib seaborn pandas tqdm imageio
pip install rsl-rl-lib==2.2.4  # Go2ポリシー評価用
```

## 使用方法

### 1. 基本的な地形可視化

単一の地形を可視化して確認：

```bash
# 段差地形（レベル2）を表示
python terrain_evaluation.py --terrain_type steps --difficulty 2

# 傾斜地形（レベル3）を表示
python terrain_evaluation.py --terrain_type slopes --difficulty 3

# 全レベルを順次表示
python terrain_evaluation.py --terrain_type steps --all_levels
```

### 2. ポリシー評価

学習済みポリシーを指定した地形で評価：

```bash
# 単一レベルでの評価
python terrain_evaluation.py \
    --terrain_type steps \
    --difficulty 2 \
    --policy_path logs/go2-walking/model_100.pt \
    --config_path logs/go2-walking/cfgs.pkl \
    --max_steps 2000

# 全レベルでの評価
python terrain_evaluation.py \
    --terrain_type steps \
    --all_levels \
    --policy_path logs/go2-walking/model_100.pt \
    --config_path logs/go2-walking/cfgs.pkl
```

### 3. 包括的ベンチマーク

複数ポリシーの詳細比較：

```bash
python policy_terrain_benchmark.py \
    --policies logs/policy1/model_100.pt logs/policy2/model_100.pt \
    --configs logs/policy1/cfgs.pkl logs/policy2/cfgs.pkl \
    --names "Policy_A" "Policy_B" \
    --terrain_types steps slopes \
    --difficulties 1 2 3 \
    --runs 5 \
    --max_steps 2000
```

### 4. 結果可視化

ベンチマーク結果の可視化レポート生成：

```bash
python visualization_tools.py \
    --results_file terrain_benchmark_results/benchmark_results_20240101_120000.json \
    --output_dir visualization_output
```

## 地形設定詳細

### 段差地形（Steps）

| レベル | 段差高さ | 段差幅 | 説明 |
|--------|----------|--------|------|
| 1 | 5cm | 1.0m | 低い段差（容易） |
| 2 | 10cm | 0.8m | 中程度の段差（中級） |
| 3 | 15cm | 0.6m | 高い段差（困難） |

### 傾斜地形（Slopes）

| レベル | 傾斜角度 | Slope値 | 説明 |
|--------|----------|---------|------|
| 1 | 約6度 | 0.1 | 緩やかな傾斜（容易） |
| 2 | 約11度 | 0.2 | 中程度の傾斜（中級） |
| 3 | 約17度 | 0.3 | 急な傾斜（困難） |

## 評価メトリクス

### 基本メトリクス
- **成功率**: 目標距離（デフォルト8m）到達率
- **移動距離**: 実際に前進した距離
- **完了時間**: タスク完了または失敗までの時間

### 詳細メトリクス
- **平均速度**: 移動距離÷完了時間
- **軌跡効率**: 直線距離÷実際の移動軌跡長
- **安定性スコア**: 転倒時間を除いた安定歩行時間の割合
- **転倒回数**: 転倒検出回数
- **回復回数**: 転倒から回復した回数

### 失敗条件
- 高さが10cm以下に低下（致命的転倒）
- ロール/ピッチ角が60度以上（転倒）
- 5秒以上不安定状態が継続

## 出力ファイル

### ベンチマーク結果
```
terrain_benchmark_results/
├── benchmark_results_YYYYMMDD_HHMMSS.json  # 詳細な評価データ
├── benchmark_report_YYYYMMDD_HHMMSS.md     # マークダウンレポート
├── success_rate_heatmap.png                # 成功率ヒートマップ
├── performance_comparison.png              # パフォーマンス比較グラフ
├── trajectory_*.png                        # 各条件の軌跡分析
└── visualization_report.html               # HTML総合レポート
```

### 軌跡データ構造
```json
{
  "trajectory": {
    "positions": [[x, y, z], ...],      # ロボット位置の時系列
    "velocities": [[vx, vy, vz], ...],  # 速度ベクトルの時系列
    "orientations": [[qx, qy, qz, qw], ...], # 四元数の時系列
    "timestamps": [t1, t2, ...]         # タイムスタンプ
  }
}
```

## カスタマイズ

### 地形パラメータの変更

`TerrainEvaluationEnvironment` クラスの `_get_terrain_configs()` メソッドを編集：

```python
def _get_terrain_configs(self):
    return {
        "custom_terrain": {
            1: {"parameter": value, "description": "説明"},
            # ...
        }
    }
```

### 新しい評価メトリクスの追加

`AdvancedPolicyEvaluator` クラスの `evaluate_with_metrics()` メソッドに追加：

```python
# 新しいメトリクス計算
metrics["custom_metric"] = calculate_custom_metric(trajectory_data)
```

### 可視化のカスタマイズ

`TerrainVisualizationTools` クラスに新しいプロット関数を追加：

```python
def plot_custom_analysis(self, results, save_path=None):
    # カスタム分析プロット
    pass
```

## トラブルシューティング

### よくある問題

1. **`rsl_rl` インポートエラー**
   ```bash
   pip uninstall rsl_rl
   pip install rsl-rl-lib==2.2.4
   ```

2. **GPU メモリ不足**
   - `--max_steps` を減らす
   - バッチサイズを小さくする

3. **可視化で日本語が表示されない**
   - システムに日本語フォントをインストール
   - `matplotlib` の設定を確認

### デバッグオプション

```bash
# CPUモードで実行（デバッグ用）
python terrain_evaluation.py --terrain_type steps --difficulty 1 --no_vis

# ログレベルを上げる
export GENESIS_LOG_LEVEL=DEBUG
python terrain_evaluation.py ...
```

## 実験例

### 比較実験の例

```bash
# 1. 段差地形での複数ポリシー比較
python policy_terrain_benchmark.py \
    --policies policy_flat.pt policy_rough.pt policy_adaptive.pt \
    --configs config_flat.pkl config_rough.pkl config_adaptive.pkl \
    --names "Flat_Trained" "Rough_Trained" "Adaptive" \
    --terrain_types steps \
    --difficulties 1 2 3 \
    --runs 10

# 2. 結果可視化
python visualization_tools.py \
    --results_file terrain_benchmark_results/benchmark_results_*.json

# 3. 特定の失敗ケースを詳細分析
python terrain_evaluation.py \
    --terrain_type steps --difficulty 3 \
    --policy_path policy_flat.pt \
    --config_path config_flat.pkl \
    --max_steps 3000
```

## 参考

- Genesis Documentation: https://genesis-world.readthedocs.io/
- Go2 Robot: Unitree Go2 quadruped robot
- RSL-RL: https://github.com/leggedrobotics/rsl_rl

---

**注意**: 実際のポリシー評価には、適切に訓練されたGo2ポリシーファイルと設定ファイルが必要です。`examples/locomotion/` ディレクトリの訓練スクリプトを使用してポリシーを事前に訓練してください。