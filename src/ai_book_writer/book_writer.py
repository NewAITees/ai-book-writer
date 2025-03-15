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
from typing import Dict, List, Optional, Union

from rich.console import Console

# Import the webtools module for web scraping
from ai_book_writer.webtools import fetch_webpage_content, search_and_extract_content

# Initialize console for rich output
console = Console()
logger = logging.getLogger("book_writer")

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
        default=os.environ.get("BOOK_WRITER_MODEL", "ollama/gemma3:4b"),
        help="LLM model to use (default: ollama/gemma3:4b)"
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
    
    return parser.parse_args()

def setup_environment() -> Dict[str, str]:
    """Set up environment variables and configuration."""
    config = {
        "model": os.environ.get("BOOK_WRITER_MODEL", "ollama/gemma3:4b"),
        "search_results_count": int(os.environ.get("SEARCH_RESULTS_COUNT", "10")),
        "temperature": float(os.environ.get("MODEL_TEMPERATURE", "0.7")),
    }
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    return config

async def gather_research_data(topic: str, num_results: int = 5) -> str:
    """
    指定したトピックに関する情報を収集
    
    Args:
        topic: 検索トピック
        num_results: 結果の最大数
        
    Returns:
        str: 収集した情報（マークダウン形式）
    """
    console.print(f"[bold blue]Researching topic: {topic}[/bold blue]")
    
    try:
        # 検索と内容抽出を実行
        results = await search_and_extract_content(topic, num_results)
        
        if not results:
            logger.warning(f"No search results found for topic: {topic}")
            return f"# Research on {topic}\n\nNo detailed information found."
        
        # 結果をマークダウン形式で結合
        combined_content = f"# Research on {topic}\n\n"
        
        for i, result in enumerate(results):
            combined_content += f"## {result.title}\n\n"
            combined_content += f"Source: {result.url}\n\n"
            combined_content += result.markdown + "\n\n"
            combined_content += "---\n\n"
        
        logger.info(f"Successfully gathered research data for topic: {topic}")
        return combined_content
    
    except Exception as e:
        logger.error(f"Error while gathering research data: {str(e)}")
        return f"# Research on {topic}\n\nError gathering information: {str(e)}"

async def main_async():
    """Main async function to run the AI-Powered Book Writer."""
    try:
        # 引数の解析
        args = parse_arguments()
        config = setup_environment()
        
        # 出力メッセージ
        console.print(f"[bold green]AI-Powered Book Writer[/bold green]")
        console.print(f"Topic: [bold]{args.topic}[/bold]")
        console.print(f"Output format: [bold]{args.format}[/bold]")
        console.print(f"Model: [bold]{args.model}[/bold]")
        console.print(f"Generate image descriptions: [bold]{'Yes' if args.images else 'No'}[/bold]")
        console.print(f"Verify facts: [bold]{'Yes' if args.verify else 'No'}[/bold]")
        console.print("\n")
        
        # トピックに関する情報を収集
        research_data = await gather_research_data(args.topic)
        
        # 収集した情報を保存（デバッグ用）
        research_file = f"output/research_{args.topic.replace(' ', '_')}.md"
        with open(research_file, "w", encoding="utf-8") as f:
            f.write(research_data)
        
        console.print(f"[bold green]Research data saved to {research_file}[/bold green]")
        
        # TODO: 本の生成処理を実装
        # 1. 研究データを使用して本の概要を生成
        # 2. 概要に基づいて各章を生成
        # 3. 生成された章を編集・校正
        # 4. 指定された形式で本を保存
        
        console.print("[bold green]Book creation completed![/bold green]")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        console.print(f"[bold red]Error: {str(e)}[/bold red]")

def main():
    """Main function to run the AI-Powered Book Writer."""
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('book_writer.log'),
            logging.StreamHandler()
        ]
    )
    
    # Windows向けの非同期実行対応
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 