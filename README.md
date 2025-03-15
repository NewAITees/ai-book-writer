# AI-Powered Book Writer

このプロジェクトは、CrewAIとGemma 3を利用して、指定されたトピックに関する本を自動的に作成するツールです。複数のAIエージェントが協力して、リサーチ、アウトライン作成、章の執筆、編集を行います。

## 特徴

- **複数エージェントの協力**: リサーチ、アウトライン作成、執筆、編集のための専門エージェント
- **ウェブ検索統合**: Bright Dataのプロキシを使用して、最新の情報を検索
- **柔軟な出力形式**: Markdown、HTML、PDFでの出力に対応
- **画像説明生成**: 章の内容に合わせた画像の説明を生成
- **引用と参考文献**: 情報源の適切な引用と参考文献の管理
- **エラーハンドリング**: 堅牢なエラー処理とリトライメカニズム
- **進捗表示**: リアルタイムの進捗状況表示
- **設定の柔軟性**: 環境変数やコマンドライン引数による設定

## 必要条件

- Python 3.8以上
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

3. Bright Dataアカウントのセットアップ（ウェブ検索機能に必要）:
   - [Bright Data](https://brightdata.com/)にアカウントを作成
   - SERPアカウントを設定し、認証情報を取得
   - 環境変数を設定するか、コード内の設定を更新

4. Ollamaをインストールしてモデルをダウンロード:
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
poetry run book-writer --topic "トピック" --format md --model "ollama/gemma3:4b" --images --verify
```

オプション:
- `--topic, -t`: 本のトピック（必須）
- `--format, -f`: 出力形式（`md`, `html`, `pdf`, `all`）、デフォルトは`md`
- `--model, -m`: 使用するLLMモデル、デフォルトは`ollama/gemma3:4b`
- `--images, -i`: 画像説明を生成する
- `--verify, -v`: 事実を検証する

## 環境変数

環境変数で設定をカスタマイズできます:

```bash
export BOOK_WRITER_MODEL="ollama/gemma3:4b"
export BRIGHT_DATA_HOST="brd.superproxy.io"
export BRIGHT_DATA_PORT="33335"
export BRIGHT_DATA_USERNAME="your_username"
export BRIGHT_DATA_PASSWORD="your_password"
export SEARCH_RESULTS_COUNT="10"
export MODEL_TEMPERATURE="0.7"
```

## プロジェクト構造

```
ai-book-writer/
├── pyproject.toml     # 依存関係と設定
├── src/               # ソースコード
│   └── ai_book_writer/
│       ├── __init__.py
│       └── book_writer.py  # メインスクリプト
├── README.md          # このファイル
└── output/            # 生成された本の保存先
```

## 仕組み

1. **リサーチ**: 指定されたトピックについてBright Dataのプロキシを使用してウェブ検索を実行
2. **アウトライン生成**: 収集された情報に基づいて、本の構造とチャプター概要を生成
3. **章の執筆**: 各章を並行して生成
4. **画像説明**: （オプション）各章に関連する画像の説明を生成
5. **編集**: 生成された内容全体を見直し、一貫性と正確性を向上
6. **保存**: 指定された形式で本を保存

## トラブルシューティング

- **Ollama接続エラー**: Ollamaサービスが実行中であることを確認してください
- **Bright Data認証エラー**: 認証情報が正しく設定されているか確認してください
- **メモリエラー**: 大きな本を生成する場合は、より多くのRAMを持つマシンで実行してください
- **PDFエクスポートの問題**: WeasyPrintとその依存関係が正しくインストールされているか確認してください

## ライセンス

MITライセンス

## 謝辞

- [CrewAI](https://github.com/joaomdmoura/crewAI) - マルチエージェントフレームワーク
- [Ollama](https://github.com/ollama/ollama) - ローカルLLMサービング
- [Bright Data](https://brightdata.com/) - ウェブスクレイピングと検索API
- Google DeepMind - Gemma 3モデル