# 地形評価実験 動画記録ガイド（修正版）

## 概要

Genesis v0.2.1 でのカメラ録画機能を使用して地形汎化性能評価実験の動作を動画として記録するガイドです。

## 🎥 修正された動画記録方法

### 問題の解決
- **問題**: `gs.options.RecordingOptions` が存在しない
- **解決**: Genesis のカメラ録画機能 `camera.start_recording()` / `camera.stop_recording()` を使用

### 新しい評価スクリプト
- **`terrain_evaluation_video_recorder_fixed.py`** - 修正版動画記録ツール
- Genesis のカメラ録画機能を使用
- MP4形式で高品質録画

## 📋 使用方法

### 基本的な動画記録
```bash
# シンプル歩行ポリシーで段差地形を評価・録画
python terrain_evaluation_video_recorder_fixed.py \
    --terrain steps --difficulty 2 --save_video

# 学習済みポリシーで録画
python terrain_evaluation_video_recorder_fixed.py \
    --terrain steps --difficulty 2 \
    --policy logs/go2-walking/model_100.pt \
    --save_video --video_dir experiment_videos
```

### 複数レベル一括録画
```bash
# 全段差レベルを録画
for level in 1 2 3; do
    python terrain_evaluation_video_recorder_fixed.py \
        --terrain steps --difficulty $level \
        --save_video --video_dir videos/steps_experiment
done

# 全傾斜レベルを録画
for level in 1 2 3; do
    python terrain_evaluation_video_recorder_fixed.py \
        --terrain slopes --difficulty $level \
        --save_video --video_dir videos/slopes_experiment
done
```

## 🎬 録画設定

### カメラ設定
```python
# 録画用カメラの設定
camera = scene.add_camera(
    res=(1280, 720),              # HD解像度
    pos=(4.0, 4.0, 3.0),          # 斜め上からの視点
    lookat=(0.0, 0.0, 0.5),       # ロボット中心を注視
    fov=45,                       # 視野角45度
    GUI=False,                    # GUI表示なし（録画専用）
)
```

### 録画制御
```python
# 録画開始
camera.start_recording()

# メインループ
for step in range(max_steps):
    scene.step()
    camera.render()  # フレームごとに録画

# 録画停止・保存
camera.stop_recording(save_to_filename=video_path, fps=30)
```

## 🔧 修正内容

### 1. RecordingOptions エラーの修正
```python
# 修正前（エラー）
recording_options=gs.options.RecordingOptions(...)

# 修正後（動作）
camera = scene.add_camera(...)
camera.start_recording()
```

### 2. カメラベース録画の実装
```python
def create_height_field_terrain_with_camera(terrain_type, difficulty, save_video=True):
    scene = gs.Scene(...)
    
    # 地形作成
    terrain = scene.add_entity(gs.morphs.Terrain(...))
    
    # 録画用カメラ追加
    camera = None
    if save_video:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=(4.0, 4.0, 3.0),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            GUI=False,
        )
    
    return scene, camera, video_path
```

### 3. エラーハンドリングの改善  
```python
# 録画停止時のエラーハンドリング
try:
    camera.stop_recording(save_to_filename=video_path, fps=30)
    print(f"✅ 動画保存完了: {video_path}")
except Exception as e:
    print(f"⚠ 動画保存エラー: {e}")
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

1. **カメラが作成されない**
   ```bash
   # シーン構築前にカメラを追加しているか確認
   camera = scene.add_camera(...)
   scene.build()  # この後でないとカメラが使用できない
   ```

2. **動画ファイルが空**
   ```bash
   # camera.render() がループ内で呼ばれているか確認
   for step in range(max_steps):
       scene.step()
       camera.render()  # これが重要
   ```

3. **録画が開始されない**
   ```python
   # start_recording() が呼ばれているか確認
   camera.start_recording()
   # メインループ
   camera.stop_recording(save_to_filename=video_path, fps=30)
   ```

### パフォーマンス最適化

```python
# 高品質録画のための設定
camera = scene.add_camera(
    res=(1920, 1080),  # フルHD解像度
    pos=(4.0, 4.0, 3.0),
    lookat=(0.0, 0.0, 0.5), 
    fov=45,
    GUI=False,         # GUI無効で性能向上
)

# 録画時のFPS設定
camera.stop_recording(save_to_filename=video_path, fps=60)  # 高フレームレート
```

## 🚀 実行例

### 即座に失敗動画を記録
```bash
# 現在の実験（1ステップ失敗）を動画で記録
python terrain_evaluation_video_recorder_fixed.py \
    --terrain steps --difficulty 2 \
    --save_video \
    --video_dir reports/terrain_evaluation_2024-07-31/videos

# 結果: reports/terrain_evaluation_2024-07-31/videos/steps_level2_XXXX.mp4
```

### 複数レベル比較動画
```bash
# 全段差レベルでの失敗を記録
for level in 1 2 3; do
    python terrain_evaluation_video_recorder_fixed.py \
        --terrain steps --difficulty $level \
        --save_video \
        --video_dir reports/terrain_evaluation_2024-07-31/videos/comparison
done
```

## 📊 レポートへの動画組み込み

### Markdown での動画埋め込み
```markdown
## 実験動画記録

### 主要な失敗パターン
![Go2転倒動画](videos/steps_level2_failure.mp4)

*10cm段差でのGo2ロボット転倒（1ステップで完全失敗）*

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
  Go2ロボット転倒動画
</video>
```

## 🎯 推奨録画シナリオ

### 1. 基本失敗パターン
```bash
# 現在の実験結果（1ステップで転倒）を録画
python terrain_evaluation_video_recorder_fixed.py \
    --terrain steps --difficulty 2 \
    --policy path/to/trained_model.pt \
    --save_video --video_dir reports/terrain_evaluation_2024-07-31/videos
```

### 2. 比較実験
```bash
# 学習済み vs シンプルポリシーの比較
python terrain_evaluation_video_recorder_fixed.py \
    --terrain steps --difficulty 1 \
    --policy path/to/trained_model.pt \
    --save_video --video_dir comparison_videos

python terrain_evaluation_video_recorder_fixed.py \
    --terrain steps --difficulty 1 \
    --save_video --video_dir comparison_videos
```

### 3. 段階的難易度
```bash
# 全レベルでの失敗パターンを記録
for level in 1 2 3; do
    echo "Recording level $level..."
    python terrain_evaluation_video_recorder_fixed.py \
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
# 複数アングルでの録画設定例
camera_positions = [
    (4.0, 4.0, 3.0),   # 斜め上から（全体像）
    (2.0, 0.0, 1.0),   # 側面から（歩行動作）
    (0.0, 2.0, 1.0),   # 前面から（地形との接触）
    (6.0, 6.0, 4.0),   # 遠景（軌跡確認）
]
```

---

**注意**: 
- Genesis v0.2.1 では `RecordingOptions` は使用できません
- カメラベースの録画 (`camera.start_recording()`) を使用してください
- 動画ファイルは容量が大きいため、圧縮または外部ストレージの使用を推奨します