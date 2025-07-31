# 地形汎化性能評価ツール

複雑な地形において既存ポリシーの汎化性能を評価するためのツールセットです。学習済みポリシーの地形での性能を体系的に分析できます。

## 機能概要

- **🏔️ 地形生成**: 段差・傾斜の3段階難易度設定
- **🤖 ポリシー統合**: 学習済みポリシー読み込み＋シンプル歩行フォールバック
- **📊 詳細評価**: 成功率、移動距離、安定性、軌跡分析
- **🔧 エラー対応**: インポートエラー修正とフォールバック機能
- **📋 複数ツール**: 用途別に最適化された評価スクリプト

## ツール一覧

### 1. `terrain_evaluation_with_policy.py` 【推奨】
**学習済みポリシーと統合した完全版**

```bash
# 学習済みポリシーで評価
python terrain_evaluation_with_policy.py --terrain steps --difficulty 2 \
    --policy logs/go2-walking/model_100.pt \
    --config logs/go2-walking/cfgs.pkl

# シンプル歩行ポリシーで評価（ポリシー無しでも動作）
python terrain_evaluation_with_policy.py --terrain steps --difficulty 2

# 全レベル一括評価
python terrain_evaluation_with_policy.py --terrain slopes --all_levels \
    --policy logs/model.pt --config logs/config.pkl
```

### 2. `terrain_evaluation_final.py`
**地形生成に特化したツール**

```bash
# 地形の可視化・検証用
python terrain_evaluation_final.py --terrain steps --difficulty 2

# 高さフィールド方式で生成
python terrain_evaluation_final.py --terrain slopes --difficulty 3 --height_field
```

### 3. `terrain_evaluation_standalone.py`
**Genesis不要の設定確認ツール**

```bash
# 地形設定の詳細を表示
python terrain_evaluation_standalone.py --terrain steps --all_levels --show_code

# 特定レベルの分析
python terrain_evaluation_standalone.py --terrain slopes --difficulty 3
```

### 4. `fix_import_error.sh`
**Cythonインポートエラー修正スクリプト**

```bash
# Genesis環境の修復
chmod +x fix_import_error.sh
./fix_import_error.sh
```

### 地形設定

#### 段差地形（Steps）
- レベル1: 5cm高、1.0m幅（容易）
- レベル2: 10cm高、0.8m幅（中級）
- レベル3: 15cm高、0.6m幅（困難）

#### 傾斜地形（Slopes）  
- レベル1: 6度傾斜（容易）
- レベル2: 11度傾斜（中級）
- レベル3: 17度傾斜（困難）

## 評価メトリクス

### 基本メトリクス
- **成功率**: 8m前進達成率（with_policy版）/ 5m前進達成率（その他）
- **移動距離**: 転倒・失敗までの実際の前進距離
- **完了時間**: タスク完了または失敗までのステップ数
- **安定性**: 高さ範囲（最大・最小高度）の追跡

### 詳細分析
- **ポリシータイプ**: 学習済み／シンプル歩行の識別
- **失敗原因**: 転倒（高度0.2m以下）／時間切れの分類
- **進行状況**: 200ステップ毎の位置レポート

## セットアップ

### 必須依存関係
```bash
# 基本環境
pip install genesis-world torch numpy

# 学習済みポリシー使用時（オプション）
pip install rsl-rl-lib==2.2.4
```

### Genesis環境問題の修正
```bash
# Cythonインポートエラーが発生した場合
chmod +x fix_import_error.sh
./fix_import_error.sh /path/to/genesis
```

## 実際の使用例

### 🚀 クイックスタート
```bash
# 1. 基本的な地形確認
python terrain_evaluation_standalone.py --terrain steps --all_levels

# 2. シンプル歩行で動作テスト
python terrain_evaluation_with_policy.py --terrain steps --difficulty 1

# 3. 学習済みポリシーで本格評価
python terrain_evaluation_with_policy.py --terrain steps --all_levels \
    --policy path/to/model.pt --config path/to/config.pkl
```

### 📊 比較実験の例
```bash
# 段差での性能比較
for level in 1 2 3; do
    echo "=== Level $level ==="
    python terrain_evaluation_with_policy.py \
        --terrain steps --difficulty $level \
        --policy policy_A.pt --config config_A.pkl
done

# 傾斜での全レベル一括評価
python terrain_evaluation_with_policy.py --terrain slopes --all_levels \
    --policy policy_B.pt --config config_B.pkl --max_steps 3000
```

## トラブルシューティング

### よくある問題と解決法

1. **`ImportError: cannot import name '_replay'`**
   ```bash
   ./fix_import_error.sh
   ```

2. **`Unrecognized attribute: subterrain_params`**
   ```bash
   # final版またはwith_policy版を使用
   python terrain_evaluation_with_policy.py --terrain steps --difficulty 2
   ```

3. **ロボットが歩行しない**
   ```bash
   # ポリシー統合版を使用
   python terrain_evaluation_with_policy.py --terrain steps --difficulty 1
   ```

4. **学習済みポリシーが読み込めない**
   ```bash
   # RSL-RLライブラリをインストール
   pip install rsl-rl-lib==2.2.4
   ```

## 対応ロボット・ポリシー

- **Go2 quadruped robot** (Unitree): デフォルト対応
- **RSL-RL形式のポリシー**: `.pt` + `.pkl` ファイル
- **カスタムポリシー**: SimpleWalkingPolicyクラスを参考に拡張可能

## 研究応用

このツールセットを使用することで以下の研究が可能：

- **汎化性能評価**: 平地で学習したポリシーの地形適応能力
- **失敗モード分析**: 特定の地形特徴での転倒パターン
- **ベンチマーク構築**: 標準化された地形難易度での性能比較
- **ポリシー改善**: 弱点となる地形条件の特定

---

**注意**: 実際の性能評価には適切に訓練されたGo2ポリシーが必要です。`examples/locomotion/go2_train.py`を使用して事前にポリシーを訓練することを推奨します。