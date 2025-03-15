#!/usr/bin/env python3
"""
Book formatter module to convert book content to various output formats.
Supports Markdown, HTML, and PDF outputs with customizable templates.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import markdown
import jinja2

# ロギング設定
logger = logging.getLogger("book_formatter")

# テンプレートパスを設定
TEMPLATE_DIR = os.environ.get(
    "BOOK_WRITER_TEMPLATE_DIR", 
    os.path.join(os.path.dirname(__file__), "templates")
)

# テンプレートが存在しない場合は作成
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Jinja2環境の設定
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
    autoescape=jinja2.select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True
)

# デフォルトHTMLテンプレート
DEFAULT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ book.title }}</title>
    <style>
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        h2 {
            color: #3498db;
            margin-top: 30px;
        }
        h3 {
            color: #2980b9;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        blockquote {
            border-left: 4px solid #ccc;
            padding-left: 15px;
            color: #555;
            font-style: italic;
        }
        code {
            background-color: #f9f9f9;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .chapter {
            margin-bottom: 50px;
            page-break-after: always;
        }
        .toc {
            margin: 30px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 15px;
        }
        .toc a {
            display: block;
            padding: 5px 0;
        }
        .cover {
            text-align: center;
            margin-bottom: 50px;
            page-break-after: always;
        }
        @media print {
            body {
                max-width: none;
            }
        }
    </style>
</head>
<body>
    <div class="cover">
        <h1>{{ book.title }}</h1>
        {% if book.subtitle %}
        <h2>{{ book.subtitle }}</h2>
        {% endif %}
        <p><em>{{ book.description }}</em></p>
    </div>
    
    <div class="toc">
        <h2>目次</h2>
        <ul>
            {% for chapter in book.chapters %}
            <li><a href="#chapter-{{ loop.index }}">{{ chapter.title }}</a></li>
            {% endfor %}
        </ul>
    </div>
    
    {% for chapter in book.chapters %}
    <div class="chapter" id="chapter-{{ loop.index }}">
        {{ chapter.html_content|safe }}
    </div>
    {% endfor %}
</body>
</html>
"""

def ensure_template_files() -> None:
    """必要なテンプレートファイルが存在することを確認し、存在しない場合は作成する"""
    html_template_path = os.path.join(TEMPLATE_DIR, "book_template.html")
    
    # HTMLテンプレートの作成
    if not os.path.exists(html_template_path):
        with open(html_template_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_HTML_TEMPLATE)
        logger.info(f"Created default HTML template at {html_template_path}")

def format_markdown_book(book_data: Dict[str, Any]) -> str:
    """
    本の内容をMarkdown形式に変換する
    
    Args:
        book_data: 本のデータ（タイトル、章など）
        
    Returns:
        str: Markdown形式のテキスト
    """
    try:
        # 表紙ページ
        md_content = f"# {book_data['title']}\n\n"
        
        if book_data.get('subtitle'):
            md_content += f"## {book_data['subtitle']}\n\n"
            
        md_content += f"*{book_data['description']}*\n\n"
        
        if book_data.get('target_audience'):
            md_content += f"対象読者: {book_data['target_audience']}\n\n"
            
        # 目次
        md_content += "## 目次\n\n"
        for i, chapter in enumerate(book_data['chapters']):
            md_content += f"{i+1}. [{chapter['title']}](#chapter-{i+1})\n"
        
        md_content += "\n---\n\n"
        
        # 各章の内容
        for i, chapter in enumerate(book_data['chapters']):
            md_content += f"<a id='chapter-{i+1}'></a>\n\n"
            md_content += chapter['content']
            md_content += "\n\n---\n\n"
        
        return md_content
        
    except Exception as e:
        logger.error(f"Error formatting markdown book: {str(e)}")
        return f"# {book_data.get('title', 'Error')}\n\nエラーが発生しました: {str(e)}"

def format_html_book(book_data: Dict[str, Any], template_name: str = "book_template.html") -> str:
    """
    本の内容をHTML形式に変換する
    
    Args:
        book_data: 本のデータ（タイトル、章など）
        template_name: 使用するテンプレート名
        
    Returns:
        str: HTML形式のテキスト
    """
    try:
        # テンプレートの存在を確認
        ensure_template_files()
        
        # 各章のMarkdownをHTMLに変換
        for chapter in book_data['chapters']:
            chapter['html_content'] = markdown.markdown(
                chapter['content'],
                extensions=['extra', 'toc', 'sane_lists', 'codehilite']
            )
        
        # テンプレートを取得
        template = jinja_env.get_template(template_name)
        
        # テンプレートを適用
        html_content = template.render(book=book_data)
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error formatting HTML book: {str(e)}")
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"

def save_book_to_file(content: str, output_path: str) -> bool:
    """
    本の内容をファイルに保存する
    
    Args:
        content: 保存する内容
        output_path: 出力ファイルパス
        
    Returns:
        bool: 成功した場合True
    """
    try:
        # 出力ディレクトリを確保
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # ファイルに書き込み
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Successfully saved book to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving book to file {output_path}: {str(e)}")
        return False

def convert_to_pdf(html_path: str, pdf_path: str) -> bool:
    """
    HTML形式の本をPDFに変換する（WeasyPrintを使用）
    
    Args:
        html_path: HTMLファイルパス
        pdf_path: 出力PDFファイルパス
        
    Returns:
        bool: 成功した場合True
    """
    try:
        # WeasyPrintのインポートは実行時に行う（オプション依存のため）
        try:
            from weasyprint import HTML
        except ImportError:
            logger.error("WeasyPrint is not installed. Please install with 'pip install weasyprint' or 'poetry install -E pdf'.")
            return False
        
        # HTMLファイルの存在を確認
        if not os.path.exists(html_path):
            logger.error(f"HTML file does not exist: {html_path}")
            return False
        
        # PDFに変換
        HTML(html_path).write_pdf(pdf_path)
        
        logger.info(f"Successfully converted HTML to PDF: {pdf_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting HTML to PDF: {str(e)}")
        return False

def save_book_to_formats(book_data: Dict[str, Any], topic: str, formats: List[str]) -> Dict[str, str]:
    """
    本のデータを指定された形式で保存する
    
    Args:
        book_data: 本のデータ
        topic: 本のトピック（ファイル名生成用）
        formats: 出力する形式のリスト（'md', 'html', 'pdf'）
        
    Returns:
        Dict[str, str]: 形式ごとの出力ファイルパス
    """
    # ファイル名のベース部分を生成
    base_filename = topic.replace(' ', '_').lower()
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = {}
    
    try:
        # Markdown形式
        if 'md' in formats or 'all' in formats:
            md_content = format_markdown_book(book_data)
            md_path = os.path.join(output_dir, f"{base_filename}.md")
            if save_book_to_file(md_content, md_path):
                output_files['md'] = md_path
        
        # HTML形式
        if 'html' in formats or 'pdf' in formats or 'all' in formats:
            html_content = format_html_book(book_data)
            html_path = os.path.join(output_dir, f"{base_filename}.html")
            if save_book_to_file(html_content, html_path):
                output_files['html'] = html_path
                
                # PDF形式（HTMLから変換）
                if 'pdf' in formats or 'all' in formats:
                    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
                    if convert_to_pdf(html_path, pdf_path):
                        output_files['pdf'] = pdf_path
        
        return output_files
        
    except Exception as e:
        logger.error(f"Error saving book to formats: {str(e)}")
        return output_files

# メインの部分（テスト用）
if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # テスト用のサンプルブック
    sample_book = {
        "title": "Pythonプログラミング入門",
        "subtitle": "初心者から中級者へのステップアップ",
        "description": "Pythonプログラミングについてゼロからわかりやすくまとめたガイドブックです。",
        "target_audience": "プログラミング初心者、Python初心者",
        "chapters": [
            {
                "title": "Pythonの基礎",
                "summary": "Pythonの基本的な概念と特徴について解説します。",
                "content": "# Pythonの基礎\n\nPythonは読みやすく書きやすい高水準プログラミング言語です。\n\n## Pythonの特徴\n\n- シンプルで読みやすい構文\n- 豊富なライブラリ\n- 多様なプラットフォームでの実行\n\n## インストール方法\n\nPythonは公式サイトからダウンロードできます。"
            },
            {
                "title": "データ型とコントロールフロー",
                "summary": "Pythonのデータ型と制御構造について解説します。",
                "content": "# データ型とコントロールフロー\n\nPythonには様々なデータ型があります。\n\n## 基本データ型\n\n- 数値型（int, float）\n- 文字列型（str）\n- ブール型（bool）\n\n## 制御構造\n\n```python\nif condition:\n    # do something\nelse:\n    # do something else\n```"
            }
        ],
        "estimated_pages": 120
    }
    
    # テスト関数
    def test_formatters():
        # テスト内容
        print("Testing Markdown formatter...")
        md_content = format_markdown_book(sample_book)
        print(f"Generated {len(md_content)} characters of Markdown")
        
        print("\nTesting HTML formatter...")
        html_content = format_html_book(sample_book)
        print(f"Generated {len(html_content)} characters of HTML")
        
        print("\nSaving to files...")
        output_files = save_book_to_formats(sample_book, "Python Programming", ["md", "html"])
        print(f"Output files: {output_files}")
    
    # テスト実行
    test_formatters() 