# 地形評価実験 動画記録ガイド

## 概要

地形汎化性能評価実験の動作を動画として記録し、レポートに添付するためのガイドです。

## 🎥 動画記録ツール

### 新しい評価スクリプト
- **`terrain_evaluation_video_recorder.py`** - 動画記録機能付きの評価ツール
- Genesis の内蔵録画機能を使用
- MP4形式で高品質録画

## 📋 使用方法

### 基本的な動画記録
```bash
# シンプル歩行ポリシーで段差地形を評価・録画
python terrain_evaluation_video_recorder.py \
    --terrain steps --difficulty 2 --save_video

# 学習済みポリシーで録画
python terrain_evaluation_video_recorder.py \
    --terrain steps --difficulty 2 \
    --policy logs/go2-walking/model_100.pt \
    --save_video --video_dir experiment_videos
```

### 複数レベル一括録画
```bash
# 全段差レベルを録画
for level in 1 2 3; do
    python terrain_evaluation_video_recorder.py \
        --terrain steps --difficulty $level \
        --save_video --video_dir videos/steps_experiment
done

# 全傾斜レベルを録画
for level in 1 2 3; do
    python terrain_evaluation_video_recorder.py \
        --terrain slopes --difficulty $level \
        --save_video --video_dir videos/slopes_experiment
done
```

## 🎬 録画設定

### 動画品質設定
```python
recording_options=gs.options.RecordingOptions(
    record=True,
    video_path=video_path,
    fps=30,           # 30fps で滑らかな動画
    record_rgb=True,  # RGB カラー録画
    record_depth=False, # 深度情報は不要
)
```

### カメラ設定
```python
viewer_options=gs.options.ViewerOptions(
    camera_pos=(4.0, 4.0, 3.0),    # 斜め上からの視点
    camera_lookat=(0.0, 0.0, 0.5), # ロボット中心を注視
    camera_fov=45,                 # 視野角45度
    max_FPS=50,                    # 高品質レンダリング
)
```

## 📁 動画ファイル管理

### 自動ファイル名生成
```
videos/
├── steps_level1_1690876543.mp4    # 段差レベル1
├── steps_level2_1690876600.mp4    # 段差レベル2（失敗例）
├── steps_level3_1690876657.mp4    # 段差レベル3
├── slopes_level1_1690876714.mp4   # 傾斜レベル1
├── slopes_level2_1690876771.mp4   # 傾斜レベル2
└── slopes_level3_1690876828.mp4   # 傾斜レベル3
```

### 推奨ディレクトリ構成
```
reports/terrain_evaluation_2024-07-31/
├── videos/
│   ├── failure_cases/              # 失敗例動画
│   │   ├── steps_level2_failure.mp4
│   │   └── steps_level3_immediate_fall.mp4
│   ├── comparison/                 # 比較用動画
│   │   ├── simple_policy_vs_trained.mp4
│   │   └── level1_vs_level2_vs_level3.mp4
│   └── analysis/                   # 分析用動画
│       ├── slow_motion_fall.mp4
│       └── multiple_angles.mp4
├── TERRAIN_EVALUATION_REPORT.md
├── EXPERIMENTAL_RESULTS.json
└── VIDEO_ANALYSIS.md
```

## 🔧 トラブルシューティング

### よくある問題

1. **録画が開始されない**
   ```bash
   # GenesisのRecordingOptionsが正しく設定されているか確認
   python -c "import genesis as gs; print(gs.options.RecordingOptions.__doc__)"
   ```

2. **動画ファイルが空**
   ```bash
   # 十分な実行時間があるか確認（最低100ステップ）
   python terrain_evaluation_video_recorder.py --max_steps 500 --save_video
   ```

3. **録画品質が低い**
   ```python
   # viewer_options で max_FPS を上げる
   viewer_options=gs.options.ViewerOptions(max_FPS=60)
   ```

### パフォーマンス最適化

```python
# 高品質録画のための設定
viewer_options=gs.options.ViewerOptions(
    camera_pos=(4.0, 4.0, 3.0),
    camera_lookat=(0.0, 0.0, 0.5), 
    camera_fov=45,
    max_FPS=60,        # 高フレームレート
    render_dt=0.02,    # シミュレーション同期
)

recording_options=gs.options.RecordingOptions(
    record=True,
    video_path=video_path,
    fps=30,            # 標準フレームレート
    record_rgb=True,
    record_depth=False,
    compress=True,     # ファイルサイズ圧縮
)
```

## 📊 レポートへの動画組み込み

### Markdown での動画埋め込み
```markdown
## 実験動画

### 段差地形での失敗例
![失敗動画](videos/failure_cases/steps_level2_failure.mp4)

*Go2ロボットが10cm段差で1ステップ目に転倒する様子*

### 地形レベル比較
| レベル | 動画 | 結果 |
|--------|------|------|
| 1 (5cm) | [動画](videos/steps_level1.mp4) | 成功 |
| 2 (10cm) | [動画](videos/steps_level2.mp4) | **失敗** |
| 3 (15cm) | [動画](videos/steps_level3.mp4) | 失敗 |
```

### HTML での動画埋め込み
```html
<video width="640" height="480" controls>
  <source src="videos/steps_level2_failure.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

## 🎯 推奨録画シナリオ

### 1. 基本失敗パターン
```bash
# 現在の実験結果（1ステップで転倒）を録画
python terrain_evaluation_video_recorder.py \
    --terrain steps --difficulty 2 \
    --policy path/to/trained_model.pt \
    --save_video --video_dir reports/terrain_evaluation_2024-07-31/videos
```

### 2. 比較実験
```bash
# 学習済み vs シンプルポリシーの比較
python terrain_evaluation_video_recorder.py \
    --terrain steps --difficulty 1 \
    --policy path/to/trained_model.pt \
    --save_video --video_dir comparison_videos

python terrain_evaluation_video_recorder.py \
    --terrain steps --difficulty 1 \
    --save_video --video_dir comparison_videos
```

### 3. 段階的難易度
```bash
# 全レベルでの失敗パターンを記録
for level in 1 2 3; do
    echo "Recording level $level..."
    python terrain_evaluation_video_recorder.py \
        --terrain steps --difficulty $level \
        --policy path/to/trained_model.pt \
        --save_video --video_dir difficulty_progression
done
```

## 📈 動画分析のポイント

### 記録すべき要素
1. **初期姿勢**: ロボットの地形への配置
2. **第1ステップ**: 制御指令の初回実行
3. **転倒過程**: 失敗に至るメカニズム
4. **地形との相互作用**: 接触・衝突の様子

### 分析用カメラアングル
```python
# 複数アングルでの録画
camera_positions = [
    (4.0, 4.0, 3.0),   # 斜め上から（全体像）
    (2.0, 0.0, 1.0),   # 側面から（歩行動作）
    (0.0, 2.0, 1.0),   # 前面から（地形との接触）
    (6.0, 6.0, 4.0),   # 遠景（軌跡確認）
]
```

---

**注意**: 動画ファイルは容量が大きいため、GitHub等にアップロードする際は圧縮または外部ストレージ（YouTube、Google Drive等）の使用を推奨します。