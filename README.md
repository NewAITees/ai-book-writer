I'll update the README to include more detailed information about how to use the book writing functionality and the available options. Here's an improved README with more comprehensive usage instructions:

# AI-Powered Book Writer

AIパワードブックライターは、OllamaとGemma 3モデルを利用して、指定されたトピックに関する本を自動的に作成するツールです。

## 特徴

- **AIによる本の生成**: 指定したトピックについて、リサーチ、アウトライン作成、章の執筆を自動化
- **ウェブ検索統合**: Crawl4AIを使用して、トピックに関する最新の情報を検索
- **柔軟な出力形式**: Markdown、HTML、PDFでの出力に対応
- **画像説明生成**: 章の内容に合わせた画像の説明を生成（オプション）
- **ファクトチェック**: 生成された内容の事実確認（オプション）
- **柔軟な設定**: コマンドライン引数や環境変数による細かな設定

## 必要条件

- Python 3.10以上
- OllamaがインストールされていてGemma 3モデルが利用可能であること
- （オプション）PDFエクスポートにはWeasyPrintが必要

## インストール

1. リポジトリをクローン:
   ```bash
   git clone https://github.com/yourusername/ai-book-writer.git
   cd ai-book-writer
   ```

2. Poetryで環境をセットアップ:
   ```bash
   # Poetryがインストールされていない場合
   curl -sSL https://install.python-poetry.org | python3 -

   # 依存関係のインストール
   poetry install

   # PDFエクスポートを使用する場合
   poetry install -E pdf
   ```

3. Ollamaをインストールしてモデルをダウンロード:
   ```bash
   # Ollamaのインストール（プラットフォームに応じた方法で）
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Gemma 3モデルのダウンロード
   ollama pull gemma3:4b
   ```

## 使用方法

### 基本的な使用方法

最も基本的な形式では、トピックのみを指定して本を生成できます：

```bash
poetry run book-writer --topic "人工知能の歴史と未来"
```

これは指定したトピックに関する本を生成し、デフォルトでMarkdown形式で保存します。

### 全オプション

```bash
poetry run book-writer --topic "トピック" --format [md|html|pdf|all] --model "gemma3:4b" --images --verify --results 10 --temperature 0.7
```

#### 主要オプション:
- `--topic, -t`: 本のトピック（必須）
- `--format, -f`: 出力形式（`md`, `html`, `pdf`, `all`）、デフォルトは`md`
- `--model, -m`: 使用するOllamaモデル、デフォルトは`gemma3:4b`
- `--images, -i`: 画像説明を生成する（フラグ）
- `--verify, -v`: 事実を検証する（フラグ）
- `--results, -r`: 取得する検索結果の数（デフォルト: 5）
  - 値を増やすと、より多くの情報を収集できますが、処理時間が長くなります
  - 例: `--results 10`（10件の検索結果を使用）
- `--temperature`: テキスト生成の温度パラメータ（デフォルト: 0.7）
  - 低い値（例: 0.3）: より事実に基づいた一貫性のある出力
  - 高い値（例: 0.9）: より創造的で多様な出力

### 例

詳細な機械学習に関する本をHTML形式で作成し、より多くの検索結果を使用：
```bash
poetry run book-writer --topic "機械学習の基礎から応用まで" --format html --results 15 --temperature 0.6
```

料理に関する本をすべての形式で生成し、画像説明を含める：
```bash
poetry run book-writer --topic "日本の伝統料理" --format all --images --results 12
```

より大きなモデルを使って科学の本を作成：
```bash
poetry run book-writer --topic "宇宙物理学の最新発見" --model "gemma3:27b" --verify

poetry run book-writer --topic "ollama の様々な使い方" --model "gemma3:27b" --verify --results 15 --temperature 0.6

```

## 環境変数によるカスタマイズ

以下の環境変数を設定することで、デフォルト設定をカスタマイズできます：

```bash
# 使用するモデル
export BOOK_WRITER_MODEL="gemma3:4b"

# 検索結果の数
export SEARCH_RESULTS_COUNT="10"

# 生成温度
export MODEL_TEMPERATURE="0.7"

# テンプレートディレクトリの場所
export BOOK_WRITER_TEMPLATE_DIR="/path/to/your/templates"
```

## パフォーマンスとリソース使用量の調整

1. **検索結果数の調整**:
   - `--results`パラメータを増やすと情報の質が向上しますが、処理時間が長くなります
   - リソースに制約がある場合は5〜10の値を推奨、高性能な環境では15〜20も使用可能

2. **モデルサイズの選択**:
   - 小型モデル（`gemma3:4b`）: 処理が速く、基本的な本の生成に適しています
   - 大型モデル（`gemma3:27b`）: 高品質な出力が得られますが、より多くのRAMとGPUリソースが必要

3. **温度パラメータの調整**:
   - 事実に基づいた技術書: 0.3〜0.5の低い温度を推奨
   - 創造的な内容: 0.7〜0.9の高い温度を使用

## 出力形式

プログラムは以下の形式で本を出力できます：

1. **Markdown (md)**: 最も軽量で編集しやすい形式
2. **HTML (html)**: ウェブブラウザで閲覧可能な形式
3. **PDF (pdf)**: 印刷や配布に適した固定レイアウト形式
4. **全形式 (all)**: 上記すべての形式を一度に生成

出力ファイルは`output/`ディレクトリに保存されます。

## プロジェクト構造

```
ai-book-writer/
├── pyproject.toml     # 依存関係と設定
├── src/               # ソースコード
│   └── ai_book_writer/
│       ├── __init__.py
│       ├── book_writer.py      # メインスクリプト
│       ├── ollama_helpers.py   # Ollama API操作ヘルパー
│       ├── webtools.py         # Web検索・スクレイピング機能
│       ├── book_formatter.py   # 出力フォーマット処理
│       └── templates/          # テンプレートディレクトリ
├── README.md          # このファイル
└── output/            # 生成された本の保存先
```

## 仕組み

1. **リサーチ**: 指定されたトピックについてCrawl4AIを使用してウェブ検索を実行
   - 検索結果数は`--results`オプションでカスタマイズ可能
   - 各ページの内容を抽出し、構造化されたデータとして保存

2. **アウトライン生成**: 収集された情報に基づいて、本の構造とチャプター概要を生成
   - タイトル、サブタイトル、章立て、対象読者などを決定
   - 各章の目的と概要をAIが計画

3. **章の執筆**: 各章の内容を生成
   - 章ごとにリサーチデータを参照して内容を作成
   - `--temperature`パラメータで創造性のレベルを調整可能

4. **画像説明**: （`--images`オプション有効時）各章に関連する画像の説明を生成
   - 内容を視覚的に補完するための画像コンセプトを提案

5. **ファクトチェック**: （`--verify`オプション有効時）生成された内容の事実確認を実施

6. **編集と保存**: 生成された内容全体を整形し、指定された形式で保存
   - Markdown、HTML、PDF形式に対応
   - テンプレートはカスタマイズ可能

## トラブルシューティング

- **Ollama接続エラー**: 
  - Ollamaサービスが実行中であることを確認してください
  - `ollama list`でモデルがインストールされていることを確認

- **メモリエラー**: 
  - 大きな本を生成する場合は、より多くのRAMを持つマシンで実行
  - `--results`の値を下げて使用するリソースを削減

- **PDFエクスポートの問題**: 
  - WeasyPrintとその依存関係が正しくインストールされているか確認
  - `poetry install -E pdf`コマンドで必要なパッケージをインストール

- **処理速度が遅い**:
  - 小さいモデル（`gemma3:4b`）を使用
  - 検索結果数（`--results`）を減らす
  - GPUが利用可能な場合は、OllamaがGPU加速を使用していることを確認

## ライセンス

MITライセンス