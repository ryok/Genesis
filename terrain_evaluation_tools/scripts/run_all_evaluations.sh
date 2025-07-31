#!/bin/bash

# 地形汎化性能評価 - 包括的評価スクリプト
# 全地形タイプ・全難易度レベルでの評価を自動実行

set -e

# 設定
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_ROOT="$(dirname "$SCRIPTS_DIR")"
OUTPUT_DIR="$TOOLS_ROOT/outputs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"/{videos,reports,data}

echo "================================================================"
echo "地形汎化性能評価 - 包括的バッチ実行"
echo "開始時刻: $(date)"
echo "出力先: $OUTPUT_DIR"
echo "================================================================"
echo

# 関数定義
log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

run_evaluation() {
    local terrain=$1
    local difficulty=$2
    local policy_path=${3:-""}
    
    local desc_steps=("低い段差(5cm)" "中程度の段差(10cm)" "高い段差(15cm)")
    local desc_slopes=("緩やかな傾斜(6度)" "中程度の傾斜(11度)" "急な傾斜(17度)")
    
    if [ "$terrain" = "steps" ]; then
        local description=${desc_steps[$((difficulty-1))]}
    else
        local description=${desc_slopes[$((difficulty-1))]}
    fi
    
    log_info "評価開始: $terrain レベル$difficulty - $description"
    
    local cmd_args="--terrain $terrain --difficulty $difficulty"
    if [ -n "$policy_path" ] && [ -f "$policy_path" ]; then
        cmd_args="$cmd_args --policy $policy_path"
        log_info "学習済みポリシーを使用: $policy_path"
    else
        log_info "シンプル歩行ポリシーを使用"
    fi
    
    # 基本評価実行
    local log_file="$OUTPUT_DIR/data/${terrain}_level${difficulty}_${TIMESTAMP}.log"
    if python "$SCRIPTS_DIR/terrain_evaluation_final.py" $cmd_args > "$log_file" 2>&1; then
        log_info "評価完了: $terrain レベル$difficulty"
    else
        log_error "評価失敗: $terrain レベル$difficulty"
        echo "詳細は $log_file を確認してください"
    fi
}

run_video_evaluation() {
    local terrain=$1
    local difficulty=$2
    local policy_path=${3:-""}
    
    log_info "動画記録評価開始: $terrain レベル$difficulty"
    
    local video_dir="$OUTPUT_DIR/videos/${terrain}_analysis"
    mkdir -p "$video_dir"
    
    local cmd_args="--terrain $terrain --difficulty $difficulty --save_video --video_dir $video_dir"
    if [ -n "$policy_path" ] && [ -f "$policy_path" ]; then
        cmd_args="$cmd_args --policy $policy_path"
    fi
    
    if python "$SCRIPTS_DIR/terrain_evaluation_video_recorder_fixed.py" $cmd_args; then
        log_info "動画記録完了: $terrain レベル$difficulty"
    else
        log_error "動画記録失敗: $terrain レベル$difficulty"
    fi
}

run_visualization() {
    local terrain=$1
    local difficulty=$2
    
    log_info "地形可視化開始: $terrain レベル$difficulty"
    
    local cmd_args="--terrain $terrain --difficulty $difficulty --save_video --max_steps 500"
    
    if python "$SCRIPTS_DIR/terrain_evaluation_visualization_enhanced.py" $cmd_args; then
        log_info "地形可視化完了: $terrain レベル$difficulty"
    else
        log_error "地形可視化失敗: $terrain レベル$difficulty"
    fi
}

# メイン処理開始
echo "1. 基本評価（全地形・全レベル）"
echo "================================"

# 段差地形評価
echo
log_info "段差地形評価開始"
for level in 1 2 3; do
    run_evaluation "steps" "$level" "$POLICY_PATH"
    sleep 2  # 安定化のための待機
done

echo
log_info "傾斜地形評価開始"  
for level in 1 2 3; do
    run_evaluation "slopes" "$level" "$POLICY_PATH"
    sleep 2
done

echo
echo "2. 動画記録（重要なケース）"
echo "=========================="

# 代表的なケースを動画記録
log_info "代表ケースの動画記録開始"

# 段差レベル2（中程度・失敗が予想される）
run_video_evaluation "steps" "2" "$POLICY_PATH"

# 傾斜レベル1（成功が期待される）
run_video_evaluation "slopes" "1" "$POLICY_PATH"

# 段差レベル3（高難易度・即座に失敗が予想される）
run_video_evaluation "steps" "3" "$POLICY_PATH"

echo
echo "3. 地形可視化（デバッグ用）"
echo "=========================="

# 地形生成確認用の可視化
log_info "地形可視化開始"

run_visualization "steps" "2"
run_visualization "slopes" "2"

echo
echo "4. 結果レポート生成"
echo "=================="

# 簡易レポート生成
REPORT_FILE="$OUTPUT_DIR/reports/batch_evaluation_report_$TIMESTAMP.md"

cat > "$REPORT_FILE" << EOF
# 地形汎化性能評価 バッチ実行レポート

**実行日時**: $(date)
**Genesis バージョン**: v0.2.1
**実験対象**: Go2 四足歩行ロボット

## 実行サマリー

### 評価した地形
- 段差地形: レベル1(5cm), レベル2(10cm), レベル3(15cm)
- 傾斜地形: レベル1(6°), レベル2(11°), レベル3(17°)

### 使用ポリシー
EOF

if [ -n "$POLICY_PATH" ] && [ -f "$POLICY_PATH" ]; then
    echo "- 学習済みポリシー: \`$POLICY_PATH\`" >> "$REPORT_FILE"
else
    echo "- シンプル歩行ポリシー（デフォルト）" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

### 出力ファイル
- 評価ログ: \`outputs/data/*_$TIMESTAMP.log\`
- 動画記録: \`outputs/videos/*/\`
- 可視化動画: \`terrain_visualization/\`

### 詳細結果
個別の評価結果は各ログファイルを参照してください。

---
*このレポートは run_all_evaluations.sh により自動生成されました*
EOF

log_info "レポート生成完了: $REPORT_FILE"

echo
echo "================================================================"
echo "バッチ評価実行完了"
echo "終了時刻: $(date)"
echo "================================================================"
echo
echo "結果確認:"
echo "- ログファイル: $OUTPUT_DIR/data/"
echo "- 動画ファイル: $OUTPUT_DIR/videos/"
echo "- レポート: $REPORT_FILE"
echo
echo "次のステップ:"
echo "1. ログファイルで詳細結果を確認"
echo "2. 動画で失敗パターンを分析"
echo "3. 必要に応じて設定を調整して再実行"