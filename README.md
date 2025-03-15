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

基本的な使用方法:

```bash
poetry run book-writer --topic "人工知能の歴史と未来"
```

すべてのオプション:

```bash
poetry run book-writer --topic "トピック" --format md --model "gemma3:4b" --images --verify --results 5 --temperature 0.7
```

オプション:
- `--topic, -t`: 本のトピック（必須）
- `--format, -f`: 出力形式（`md`, `html`, `pdf`, `all`）、デフォルトは`md`
- `--model, -m`: 使用するOllamaモデル、デフォルトは`gemma3:4b`
- `--images, -i`: 画像説明を生成する
- `--verify, -v`: 事実を検証する
- `--results, -r`: 取得する検索結果の数（デフォルト: 5）
- `--temperature`: テキスト生成の温度パラメータ（デフォルト: 0.7）

## 環境変数

環境変数で設定をカスタマイズできます:

```bash
export BOOK_WRITER_MODEL="gemma3:4b"
export SEARCH_RESULTS_COUNT="5"
export MODEL_TEMPERATURE="0.7"
export BOOK_WRITER_TEMPLATE_DIR="/path/to/your/templates"
```

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
2. **アウトライン生成**: 収集された情報に基づいて、本の構造とチャプター概要を生成
3. **章の執筆**: 各章の内容を生成
4. **画像説明**: （オプション）各章に関連する画像の説明を生成
5. **編集と保存**: 生成された内容全体を整形し、指定された形式で保存

## トラブルシューティング

- **Ollama接続エラー**: Ollamaサービスが実行中であることを確認してください
- **メモリエラー**: 大きな本を生成する場合は、より多くのRAMを持つマシンで実行してください
- **PDFエクスポートの問題**: WeasyPrintとその依存関係が正しくインストールされているか確認してください

## ライセンス

MITライセンス