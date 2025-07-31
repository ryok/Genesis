# 地形汎化性能評価ツール集

Go2ロボットの地形汎化性能を評価するための統合ツールセットです。

## 📁 ディレクトリ構成

```
terrain_evaluation_tools/
├── README.md                   # このファイル
├── scripts/                    # 実行スクリプト
│   ├── terrain_evaluation_final.py                    # 最終版評価ツール
│   ├── terrain_evaluation_video_recorder_fixed.py     # 動画記録ツール
│   ├── terrain_evaluation_visualization_enhanced.py   # 強化版可視化ツール
│   ├── terrain_evaluation_video_debug.py              # デバッグ用ツール
│   └── [その他のスクリプト]
├── configs/                    # 設定ファイル
├── docs/                       # ドキュメント
│   └── TERRAIN_TOOLS_README.md
├── outputs/                    # 出力ファイル
│   ├── videos/                 # 動画ファイル
│   ├── reports/                # レポート
│   └── data/                   # 実験データ
└── examples/                   # 使用例
```

## 🚀 クイックスタート

### 基本評価
```bash
cd terrain_evaluation_tools/scripts
python terrain_evaluation_final.py --terrain steps --difficulty 2
```

### 動画記録付き評価
```bash
cd terrain_evaluation_tools/scripts
python terrain_evaluation_video_recorder_fixed.py \
    --terrain steps --difficulty 2 --save_video \
    --video_dir ../outputs/videos
```

### 地形可視化確認
```bash
cd terrain_evaluation_tools/scripts
python terrain_evaluation_visualization_enhanced.py \
    --terrain steps --difficulty 2 --save_video
```

## 📊 主要ツール

### 1. 評価スクリプト

| スクリプト | 用途 | 特徴 |
|-----------|------|------|
| `terrain_evaluation_final.py` | 基本評価 | 学習済みポリシーでの性能評価 |
| `terrain_evaluation_video_recorder_fixed.py` | 動画記録 | カメラ録画機能付き評価 |
| `terrain_evaluation_visualization_enhanced.py` | 可視化 | 地形変化の明確な可視化 |
| `terrain_evaluation_video_debug.py` | デバッグ | 回転カメラでの詳細観察 |

### 2. 地形タイプ

- **段差 (steps)**: 5cm, 10cm, 15cm の段差地形
- **傾斜 (slopes)**: 6°, 11°, 17° の傾斜地形

### 3. 難易度レベル

1. **レベル1**: 低難易度 (段差5cm / 傾斜6°)
2. **レベル2**: 中難易度 (段差10cm / 傾斜11°)
3. **レベル3**: 高難易度 (段差15cm / 傾斜17°)

## 🎯 使用シナリオ

### シナリオ1: 基本性能評価
```bash
# 全レベルでの性能評価
for level in 1 2 3; do
    python scripts/terrain_evaluation_final.py \
        --terrain steps --difficulty $level \
        --policy path/to/model.pt
done
```

### シナリオ2: 動画記録による詳細分析
```bash
# 失敗パターンの動画記録
python scripts/terrain_evaluation_video_recorder_fixed.py \
    --terrain steps --difficulty 2 \
    --policy path/to/model.pt \
    --save_video --video_dir outputs/videos/failure_analysis
```

### シナリオ3: 地形生成の検証
```bash
# 地形が正しく生成されているか確認
python scripts/terrain_evaluation_visualization_enhanced.py \
    --terrain steps --all_levels --save_video
```

## 📈 実験結果の管理

### 出力ファイル構成
```
outputs/
├── videos/
│   ├── steps_level1_success.mp4
│   ├── steps_level2_failure.mp4
│   └── slopes_level3_immediate_fall.mp4
├── reports/
│   ├── TERRAIN_EVALUATION_REPORT.md
│   ├── EXPERIMENTAL_RESULTS.json
│   └── VIDEO_RECORDING_GUIDE_FIXED.md
└── data/
    ├── evaluation_results_YYYYMMDD.json
    └── performance_metrics.csv
```

## 🔧 環境要件

- **Genesis**: v0.2.1+
- **Python**: 3.10-3.12
- **PyTorch**: 学習済みポリシー読み込み用
- **CUDA**: GPU加速推奨

## 📝 設定カスタマイズ

### 地形パラメータ調整
```python
# scripts/terrain_evaluation_final.py 内で調整可能
step_heights = {1: 5.0, 2: 10.0, 3: 15.0}  # cm
slope_angles = {1: 6, 2: 11, 3: 17}         # 度
```

### カメラ設定調整
```python
# 動画記録時のカメラ設定
camera = scene.add_camera(
    res=(1920, 1080),           # 解像度
    pos=(4.0, 4.0, 3.0),        # カメラ位置
    lookat=(0.0, 0.0, 0.5),     # 注視点
    fov=45,                     # 視野角
)
```

## 🐛 トラブルシューティング

### よくある問題

1. **地形が見えない**
   ```bash
   # 強化版可視化ツールを使用
   python scripts/terrain_evaluation_visualization_enhanced.py --terrain steps --difficulty 2
   ```

2. **動画記録エラー**
   ```bash
   # Genesis v0.2.1対応版を使用
   python scripts/terrain_evaluation_video_recorder_fixed.py --save_video
   ```

3. **ポリシー読み込みエラー**
   ```bash
   # ポリシーなしで基本動作確認
   python scripts/terrain_evaluation_final.py --terrain steps --difficulty 1
   ```

## 📊 実験レポート例

現在の実験結果（Go2ロボット）:
- **段差5cm**: 成功率 0% (1ステップで転倒)
- **段差10cm**: 成功率 0% (即座に転倒)
- **段差15cm**: 成功率 0% (即座に転倒)

**原因分析**: 
- 学習時は平面(plane.urdf)のみ使用
- 地形汎化能力が不足
- より複雑な地形での学習が必要

## 🔗 関連ファイル

- `examples/locomotion/go2_train.py` - 元の学習スクリプト
- `genesis/utils/terrain.py` - Genesis地形ユーティリティ
- `genesis/ext/isaacgym/terrain_utils.py` - IsaacGym地形ツール

## 🤝 コントリビューション

1. 新しい地形タイプの追加
2. 評価指標の改善
3. 可視化機能の強化
4. パフォーマンス最適化

---

**注意**: このツールセットはGenesis v0.2.1で動作確認済みです。新しいバージョンでは一部の機能が変更される可能性があります。