#!/usr/bin/env python3
"""
AI-Powered Book Writer

This script uses Crawl4AI and Gemma 3 to automatically generate books
on specified topics.
"""

import argparse
import os
import sys
import asyncio
import logging
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.logging import RichHandler

# Import the project modules
from ai_book_writer.webtools import (
    search_and_extract,
    fetch_webpage_content_sync as fetch_webpage_content
)
from ai_book_writer.ollama_helpers import OllamaConfig, check_model_availability, generate_book_outline, generate_chapter_content
from ai_book_writer.book_formatter import save_book_to_formats

# Initialize console for rich output
console = Console()
logger = logging.getLogger("book_writer")

# Define book types
class BookType(str, Enum):
    """本のタイプを定義する列挙型"""
    BLOG = "blog"
    HORROR = "horror"
    SCIFI = "scifi"
    PROGRAMMING = "programming"
    TECHNICAL = "technical"

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Book Writer using Crawl4AI and Gemma 3"
    )
    
    parser.add_argument(
        "--topic", "-t", 
        type=str, 
        required=True,
        help="Topic for the book"
    )
    
    parser.add_argument(
        "--format", "-f", 
        type=str, 
        choices=["md", "html", "pdf", "all"], 
        default="md",
        help="Output format (default: md)"
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default=os.environ.get("BOOK_WRITER_MODEL", "gemma3:4b"),
        help="LLM model to use (default: ollama/gemma3:4b)"
    )
    
    parser.add_argument(
        "--type", "-ty",
        type=str,
        choices=[t.value for t in BookType],
        default=BookType.TECHNICAL.value,
        help="Type of book to generate (default: technical)"
    )
    
    parser.add_argument(
        "--images", "-i", 
        action="store_true",
        help="Generate image descriptions"
    )
    
    parser.add_argument(
        "--verify", "-v", 
        action="store_true",
        help="Verify facts"
    )
    
    parser.add_argument(
        "--results", "-r",
        type=int,
        default=int(os.environ.get("SEARCH_RESULTS_COUNT", "5")),
        help="Number of search results to use (default: 5)"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float,
        default=float(os.environ.get("MODEL_TEMPERATURE", "0.7")),
        help="Temperature for text generation (default: 0.7)"
    )
    
    return parser.parse_args()

def setup_environment() -> Dict[str, str]:
    """Set up environment variables and configuration."""
    config = {
        "model": os.environ.get("BOOK_WRITER_MODEL", "gemma3:4b"),
        "search_results_count": int(os.environ.get("SEARCH_RESULTS_COUNT", "5")),
        "temperature": float(os.environ.get("MODEL_TEMPERATURE", "0.7")),
    }
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    return config

async def gather_research_data(topic: str, num_results: int = 5, book_type: str = BookType.TECHNICAL.value) -> str:
    """
    指定したトピックに関する情報を収集
    
    Args:
        topic: 検索トピック
        num_results: 結果の最大数
        book_type: 本のタイプ
        
    Returns:
        str: 収集した情報（マークダウン形式）
    """
    # 本のタイプに基づいて検索キーワードを調整
    search_modifiers = {
        BookType.BLOG.value: "blog articles",
        BookType.HORROR.value: "horror stories themes elements",
        BookType.SCIFI.value: "science fiction concepts themes",
        BookType.PROGRAMMING.value: "programming tutorial examples",
        BookType.TECHNICAL.value: "technical analysis guide"
    }
    
    # 検索クエリの調整
    search_query = topic
    if book_type in search_modifiers:
        search_query = f"{topic} {search_modifiers[book_type]}"
    
    console.print(f"[bold blue]Researching topic: {search_query}[/bold blue]")
    
    try:
        # 検索と内容抽出を実行
        results = await search_and_extract(search_query, num_results)
        
        if not results:
            logger.warning(f"No search results found for topic: {search_query}")
            return f"# Research on {topic}\n\nNo detailed information found."
        
        # 結果をマークダウン形式で結合
        combined_content = f"# Research on {topic}\n\n"
        
        for i, result in enumerate(results):
            combined_content += f"## {result.title}\n\n"
            combined_content += f"Source: {result.url}\n\n"
            combined_content += result.content + "\n\n"
            combined_content += "---\n\n"
        
        logger.info(f"Successfully gathered research data for topic: {topic}")
        return combined_content
    
    except Exception as e:
        logger.error(f"Error while gathering research data: {str(e)}")
        return f"# Research on {topic}\n\nError gathering information: {str(e)}"

async def generate_book(topic: str, research_data: str, args: argparse.Namespace, progress: Progress, task_id: TaskID, prompt_templates: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    本を生成する
    
    Args:
        topic: 本のトピック
        research_data: 収集した研究データ
        args: コマンドライン引数
        progress: 進捗表示オブジェクト
        task_id: タスクID
        prompt_templates: プロンプトテンプレート
        
    Returns:
        Dict[str, Any]: 生成された本のデータ
    """
    # Ollama設定
    ollama_config = OllamaConfig(
        model=args.model,
        temperature=args.temperature
    )
    
    # モデルの可用性をチェック
    model_available = await check_model_availability(ollama_config.model)
    
    if not model_available:
        logger.error(f"Model {ollama_config.model} is not available. Please download it with 'ollama pull {ollama_config.model}'")
        raise RuntimeError(f"Model {ollama_config.model} is not available")
    
    try:
        # 概要の生成
        progress.update(task_id, description="Generating book outline...", completed=10)
        book_outline = await generate_book_outline(
            research_data, 
            topic, 
            ollama_config
        )
        
        progress.update(task_id, description="Book outline generated", completed=20)
        logger.info(f"Generated book outline: {book_outline['title']}")
        
        # 章の数
        num_chapters = len(book_outline["chapters"])
        chapter_progress_increment = 70 / num_chapters  # 残りの70%を章の数で分割
        
        # 各章の内容を生成
        for i, chapter_info in enumerate(book_outline["chapters"]):
            progress.update(
                task_id, 
                description=f"Writing chapter {i+1}/{num_chapters}: {chapter_info['title']}...",
                completed=20 + i * chapter_progress_increment
            )
            
            # 章の内容を生成
            chapter_content = await generate_chapter_content(
                chapter_info,
                topic,
                research_data,
                book_outline,
                ollama_config
            )
            
            # 生成された内容を章に格納
            book_outline["chapters"][i]["content"] = chapter_content["content"]
            
            progress.update(
                task_id, 
                description=f"Completed chapter {i+1}/{num_chapters}",
                completed=20 + (i + 1) * chapter_progress_increment
            )
        
        # 完了
        progress.update(task_id, description="Book generation completed", completed=90)
        
        return book_outline
    
    except Exception as e:
        logger.error(f"Error generating book: {str(e)}")
        raise

def load_prompt_templates() -> Dict[str, Dict[str, str]]:
    """
    外部プロンプトファイルを読み込む
    
    Returns:
        Dict[str, Dict[str, str]]: 本のタイプごとのプロンプトテンプレート
    """
    template_dir = Path(__file__).parent / "prompts"
    
    # ディレクトリが存在しない場合は作成
    if not template_dir.exists():
        template_dir.mkdir(parents=True, exist_ok=True)
    
    # すべての本のタイプに対応するテンプレートをロード
    templates = {}
    
    for book_type in BookType:
        type_templates = {}
        type_dir = template_dir / book_type.value
        
        # タイプごとのディレクトリが存在しない場合は作成
        if not type_dir.exists():
            type_dir.mkdir(parents=True, exist_ok=True)
        
        # システムプロンプトのファイルパス
        system_path = type_dir / "system_prompt.txt"
        outline_path = type_dir / "outline_instructions.txt"
        chapter_path = type_dir / "chapter_instructions.txt"
        
        # システムプロンプトの内容をロードまたは作成
        if system_path.exists():
            with open(system_path, "r", encoding="utf-8") as f:
                type_templates["system_prompt"] = f.read()
        else:
            # デフォルトのシステムプロンプトを作成
            default_system_prompt = f"あなたは{book_type.value}の専門家です。"
            type_templates["system_prompt"] = default_system_prompt
            with open(system_path, "w", encoding="utf-8") as f:
                f.write(default_system_prompt)
        
        # 概要指示の内容をロードまたは作成
        if outline_path.exists():
            with open(outline_path, "r", encoding="utf-8") as f:
                type_templates["outline_instructions"] = f.read()
        else:
            # デフォルトの概要指示を作成
            default_outline = f"{book_type.value}として適切な本の構造を計画してください。"
            type_templates["outline_instructions"] = default_outline
            with open(outline_path, "w", encoding="utf-8") as f:
                f.write(default_outline)
        
        # 章指示の内容をロードまたは作成
        if chapter_path.exists():
            with open(chapter_path, "r", encoding="utf-8") as f:
                type_templates["chapter_instructions"] = f.read()
        else:
            # デフォルトの章指示を作成
            default_chapter = f"{book_type.value}に適した章の構成で執筆してください。"
            type_templates["chapter_instructions"] = default_chapter
            with open(chapter_path, "w", encoding="utf-8") as f:
                f.write(default_chapter)
        
        templates[book_type.value] = type_templates
    
    return templates

async def main_async():
    """Main async function to run the AI-Powered Book Writer."""
    try:
        # 引数の解析
        args = parse_arguments()
        config = setup_environment()
        
        # プロンプトテンプレートの読み込み
        prompt_templates = load_prompt_templates()
        
        # タイムスタンプ（ファイル名やログ用）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 出力メッセージ
        console.print(f"[bold green]AI-Powered Book Writer[/bold green]")
        console.print(f"Topic: [bold]{args.topic}[/bold]")
        console.print(f"Book type: [bold]{args.type}[/bold]")
        console.print(f"Output format: [bold]{args.format}[/bold]")
        console.print(f"Model: [bold]{args.model}[/bold]")
        console.print(f"Generate image descriptions: [bold]{'Yes' if args.images else 'No'}[/bold]")
        console.print(f"Verify facts: [bold]{'Yes' if args.verify else 'No'}[/bold]")
        console.print(f"Number of search results: [bold]{args.results}[/bold]")
        console.print(f"Temperature: [bold]{args.temperature}[/bold]")
        console.print("\n")
        
        # 進捗表示の設定
        with Progress() as progress:
            # タスクの作成
            overall_task = progress.add_task("[green]Writing book...", total=100)
            
            # トピックに関する情報を収集
            progress.update(overall_task, description="Researching topic...", completed=0)
            research_data = await gather_research_data(args.topic, args.results, args.type)
            
            # 収集した情報を保存（デバッグ用）
            research_file = f"output/research_{args.topic.replace(' ', '_')}_{timestamp}.md"
            with open(research_file, "w", encoding="utf-8") as f:
                f.write(research_data)
            
            progress.update(overall_task, description="Research data collected", completed=10)
            logger.info(f"Research data saved to {research_file}")
            
            # 本の生成
            book_data = await generate_book(
                args.topic, 
                research_data, 
                args, 
                progress, 
                overall_task,
                prompt_templates
            )
            
            # 生成した本のデータを保存（デバッグ用）
            book_json_file = f"output/book_data_{args.topic.replace(' ', '_')}_{timestamp}.json"
            with open(book_json_file, "w", encoding="utf-8") as f:
                json.dump(book_data, f, ensure_ascii=False, indent=2)
            
            progress.update(overall_task, description="Converting to output format(s)...", completed=90)
            
            # 本を指定された形式で保存
            formats = [args.format] if args.format != "all" else ["md", "html", "pdf"]
            output_files = save_book_to_formats(book_data, args.topic, formats)
            
            # 完了表示
            progress.update(overall_task, description="Book creation completed!", completed=100)
        
        # 出力ファイルの表示
        console.print("\n[bold green]Book creation completed![/bold green]")
        console.print("Output files:")
        for fmt, path in output_files.items():
            console.print(f"  - {fmt.upper()}: {path}")
    
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)

def main():
    """Main function to run the AI-Powered Book Writer."""
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('book_writer.log'),
            RichHandler(rich_tracebacks=True)
        ]
    )
    
    # Windows向けの非同期実行対応
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 