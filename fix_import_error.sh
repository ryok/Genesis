#!/bin/bash

echo "Genesis Cython拡張モジュールのビルドを修正します..."

# 1. 現在のディレクトリを保存
CURRENT_DIR=$(pwd)

# 2. Genesisディレクトリに移動（引数で指定可能）
GENESIS_DIR=${1:-$(pwd)}
cd "$GENESIS_DIR"

echo "作業ディレクトリ: $GENESIS_DIR"

# 3. 既存のビルドファイルをクリーンアップ
echo "既存のビルドファイルをクリーンアップ中..."
rm -rf build/
rm -rf genesis.egg-info/
find . -name "*.so" -type f -delete
find . -name "*.pyc" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# 4. Cythonと必要な依存関係をインストール
echo "必要な依存関係をインストール中..."
pip install --upgrade pip
pip install cython>=3.0.0 numpy>=1.26.4 setuptools wheel

# 5. Cython拡張モジュールをビルド
echo "Cython拡張モジュールをビルド中..."
python setup.py build_ext --inplace

# 6. Genesis を開発モードでインストール
echo "Genesisを開発モードでインストール中..."
pip install -e ".[dev]"

# 7. インストールを確認
echo "インストールを確認中..."
python -c "import genesis; print('Genesis正常にインポートできました！')"

# 元のディレクトリに戻る
cd $CURRENT_DIR

echo "完了しました！"