#!/usr/bin/env python3
"""
AI-Powered Book Writer

This script uses CrewAI and Gemma 3 to automatically generate books
on specified topics using multiple AI agents working together.
"""

import os
import json
import argparse
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import requests
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

import pydantic
from pydantic import BaseModel, Field, validator
import markdown
import ollama
from jinja2 import Environment, FileSystemLoader

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai import LLM
from crewai.flow import Flow, listen, start

# 自作ウェブ検索ツールをインポート
from ai_book_writer.webtools import DuckDuckGoSearchTool, SearchResult, fetch_webpage_content

# 設定
CONFIG = {
    "model": os.environ.get("BOOK_WRITER_MODEL", "ollama/gemma3:4b"),
    "search_results_count": int(os.environ.get("SEARCH_RESULTS_COUNT", "10")),
    "temperature": float(os.environ.get("MODEL_TEMPERATURE", "0.7")),
    "output_formats": ["md", "html", "pdf"],
    "default_output_format": "md",
    "image_generation": False,
    "verify_facts": True,
    "max_retries": 3,
    "timeout_seconds": 300,
}

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('book_writer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('book_writer')

console = Console()

# モデル定義
class WebSearchToolInput(BaseModel):
    """Input schema for WebSearchTool."""
    query: str = Field(..., description="Search query to look up information.")
    num_results: int = Field(CONFIG["search_results_count"], description="Number of results to return.")

# コンテンツモデル
class Reference(BaseModel):
    """引用情報のモデル"""
    source: str
    url: Optional[str] = None
    accessed_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

class BookSection(BaseModel):
    """本のセクション（段落、リスト項目など）"""
    content: str
    type: str = "paragraph"  # paragraph, list_item, code, quote, etc
    references: List[Reference] = []

class Chapter(BaseModel):
    """本の章"""
    title: str
    summary: str
    sections: List[BookSection] = []
    image_descriptions: List[str] = []
    
    @property
    def content(self) -> str:
        """章の内容をテキストとして返す"""
        return "\n\n".join([section.content for section in self.sections])

class Outline(BaseModel):
    """本の概要"""
    topic: str
    title: str
    subtitle: Optional[str] = None
    author: str = "AI Assistant"
    total_chapters: int
    chapter_titles: List[str]
    summary: str
    target_audience: str
    key_takeaways: List[str]
    estimated_word_count: int = 0

class BookMetadata(BaseModel):
    """本のメタデータ"""
    title: str
    subtitle: Optional[str] = None
    author: str
    date_created: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    topic: str
    summary: str
    keywords: List[str] = []
    cover_description: Optional[str] = None

class Book(BaseModel):
    """完成した本の構造"""
    metadata: BookMetadata
    outline: Outline
    chapters: List[Chapter] = []
    references: List[Reference] = []
    
    class Config:
        arbitrary_types_allowed = True

class BookState(BaseModel):
    """Flow状態管理用のモデル"""
    topic: str
    outline: Optional[Outline] = None
    chapters: List[Chapter] = []
    search_results: Dict[str, List[SearchResult]] = {}
    metadata: Optional[BookMetadata] = None
    final_book: Optional[Book] = None
    output_format: str = CONFIG["default_output_format"]

# LLMの初期化
def initialize_llm():
    """LLMモデルを初期化"""
    try:
        llm = LLM(model=CONFIG["model"], temperature=CONFIG["temperature"])
        logger.info(f"LLM initialized with model: {CONFIG['model']}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise RuntimeError(f"Could not initialize LLM: {str(e)}")

# レトライデコレータ
def retry_on_error(max_retries=CONFIG["max_retries"]):
    """関数実行をリトライするデコレータ"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        raise
                    await asyncio.sleep(2 ** attempt)  # 指数バックオフ
        return wrapper
    return decorator

# エージェント作成
def create_research_agent(llm):
    """リサーチエージェントを作成"""
    return Agent(
        role="Book Research Agent",
        goal=f"Research the topic thoroughly and collect comprehensive information about it.",
        backstory="You are an expert with a deep understanding of various subjects. "
                 "You are skilled at finding relevant information from various sources.",
        verbose=True,
        llm=llm,
        tools=[DuckDuckGoSearchTool()]
    )

def create_outline_agent(llm):
    """概要作成エージェントを作成"""
    return Agent(
        role="Book Outline Creator",
        goal="Create a detailed and well-structured outline for a book about the specified topic",
        backstory="You are an experienced book editor specialized in creating engaging "
                 "and well-structured book outlines that capture readers' attention.",
        verbose=True,
        llm=llm
    )

def create_chapter_writer_agent(llm):
    """章の執筆エージェントを作成"""
    return Agent(
        role="Chapter Writer",
        goal="Write an informative and engaging chapter for a book",
        backstory="You are a professional writer with expertise in various topics. "
                 "You excel at creating content that is both informative and engaging.",
        verbose=True,
        llm=llm
    )

def create_editor_agent(llm):
    """編集エージェントを作成"""
    return Agent(
        role="Book Editor",
        goal="Review and improve the book content for clarity, coherence, and accuracy",
        backstory="You are a senior editor with years of experience perfecting books. "
                 "You have a keen eye for detail and ensure the highest quality standards.",
        verbose=True,
        llm=llm
    )

def create_image_designer_agent(llm):
    """画像デザインエージェントを作成"""
    return Agent(
        role="Image Designer",
        goal="Create descriptions for images that will illustrate key concepts in the book",
        backstory="You are a visual designer who specializes in creating descriptive concepts "
                 "for educational and informative images that enhance written content.",
        verbose=True,
        llm=llm
    )

# タスク作成ヘルパー
def create_research_task(agent, topic):
    """リサーチタスクを作成"""
    return Task(
        description=f"Research the topic '{topic}' thoroughly and gather comprehensive information.",
        expected_output="A detailed report with key information about the topic, including facts, concepts, and sources.",
        agent=agent
    )

def create_outline_task(agent, topic, research_results):
    """概要作成タスクを作成"""
    return Task(
        description=f"Create a detailed outline for a book about '{topic}' based on the research.",
        expected_output="A complete book outline with title, chapter structure, and key points for each chapter.",
        agent=agent,
        context=f"Research results: {research_results}",
        output_pydantic=Outline
    )

def create_chapter_writing_task(agent, title, topic, outline, chapter_index):
    """章の執筆タスクを作成"""
    return Task(
        description=f"Write chapter {chapter_index+1}: '{title}' for the book about {topic}.",
        expected_output="A complete, well-written chapter with sections, examples, and proper citations.",
        agent=agent,
        context=f"This chapter is part of a book about {topic}. The outline is: {outline}",
        output_pydantic=Chapter
    )

def create_editing_task(agent, book):
    """編集タスクを作成"""
    return Task(
        description="Review and edit the entire book for consistency, clarity, and accuracy.",
        expected_output="An improved version of the book with editorial corrections and suggestions.",
        agent=agent,
        context=f"The book to edit: {json.dumps(book.dict(), indent=2)}",
        output_pydantic=Book
    )

def create_image_description_task(agent, chapter):
    """画像説明タスクを作成"""
    return Task(
        description=f"Create descriptions for images to accompany the chapter '{chapter.title}'.",
        expected_output="A list of image descriptions that would enhance the chapter content.",
        agent=agent,
        context=f"Chapter content: {chapter.content}"
    )

# Flow定義
class BookFlow(Flow[BookState]):
    """本の作成フロー"""
    
    @start()
    async def start_book_creation(self):
        """本の作成プロセスを開始"""
        logger.info(f"Starting book creation process for topic: {self.state.topic}")
        console.print(f"[bold green]Creating a book about: {self.state.topic}[/bold green]")
        
        # LLM初期化
        llm = initialize_llm()
        
        # メタデータの初期化
        self.state.metadata = BookMetadata(
            title="",  # 後で更新される
            author="AI Powered Book Writer",
            topic=self.state.topic,
            summary="",  # 後で更新される
            keywords=[]
        )
        
        # 次のステップへ
        return self.perform_research()
    
    @retry_on_error()
    async def perform_research(self):
        """トピックに関する調査を実行"""
        with console.status("[bold blue]Researching topic...[/bold blue]"):
            try:
                # リサーチエージェントを作成
                llm = initialize_llm()
                research_agent = create_research_agent(llm)
                research_task = create_research_task(research_agent, self.state.topic)
                
                # リサーチを実行
                research_crew = Crew(
                    agents=[research_agent],
                    tasks=[research_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                research_results = await research_crew.kickoff(inputs={"topic": self.state.topic})
                
                # 検索結果を保存
                self.state.search_results["main"] = research_results
                
                logger.info("Research completed successfully")
                console.print("[bold green]✓ Research completed[/bold green]")
                
                # 次のステップへ
                return self.generate_outline(research_results)
            except Exception as e:
                logger.error(f"Research failed: {str(e)}")
                console.print(f"[bold red]Research failed: {str(e)}[/bold red]")
                raise
    
    @retry_on_error()
    async def generate_outline(self, research_results):
        """本の概要を生成"""
        with console.status("[bold blue]Generating book outline...[/bold blue]"):
            try:
                # 概要作成エージェントを作成
                llm = initialize_llm()
                outline_agent = create_outline_agent(llm)
                outline_task = create_outline_task(outline_agent, self.state.topic, research_results)
                
                # 概要作成を実行
                outline_crew = Crew(
                    agents=[outline_agent],
                    tasks=[outline_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                outline_result = await outline_crew.kickoff(inputs={"topic": self.state.topic})
                
                # 結果が辞書なら、Outlineオブジェクトに変換
                if isinstance(outline_result, dict):
                    self.state.outline = Outline(**outline_result)
                else:
                    self.state.outline = outline_result
                    
                # メタデータを更新
                self.state.metadata.title = self.state.outline.title
                self.state.metadata.subtitle = self.state.outline.subtitle
                self.state.metadata.summary = self.state.outline.summary
                
                logger.info(f"Outline generated with {self.state.outline.total_chapters} chapters")
                console.print(f"[bold green]✓ Outline generated with {self.state.outline.total_chapters} chapters[/bold green]")
                
                # 本の概要を表示
                console.print("\n[bold]Book Outline:[/bold]")
                console.print(f"Title: {self.state.outline.title}")
                if self.state.outline.subtitle:
                    console.print(f"Subtitle: {self.state.outline.subtitle}")
                console.print(f"Summary: {self.state.outline.summary}")
                console.print("Chapters:")
                for i, title in enumerate(self.state.outline.chapter_titles):
                    console.print(f"  {i+1}. {title}")
                
                # 次のステップへ
                return self.generate_chapters()
            except Exception as e:
                logger.error(f"Outline generation failed: {str(e)}")
                console.print(f"[bold red]Outline generation failed: {str(e)}[/bold red]")
                raise
    
    @listen(generate_outline)
    async def generate_chapters(self):
        """各章を生成"""
        if not self.state.outline:
            raise ValueError("Outline is not generated yet")
        
        tasks = []
        llm = initialize_llm()
        chapter_writer_agent = create_chapter_writer_agent(llm)
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task_id = progress.add_task("[bold]Writing chapters...", total=len(self.state.outline.chapter_titles))
            
            # 各章を非同期で生成
            for i, title in enumerate(self.state.outline.chapter_titles):
                try:
                    chapter_task = create_chapter_writing_task(
                        chapter_writer_agent, 
                        title, 
                        self.state.topic, 
                        self.state.outline, 
                        i
                    )
                    
                    # Crewを作成して実行
                    chapter_crew = Crew(
                        agents=[chapter_writer_agent],
                        tasks=[chapter_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    # 章を生成
                    chapter_result = await chapter_crew.kickoff(
                        inputs={
                            "title": title,
                            "topic": self.state.topic,
                            "chapter_index": i,
                            "total_chapters": self.state.outline.total_chapters,
                            "outline": self.state.outline
                        }
                    )
                    
                    # 結果が辞書なら、Chapterオブジェクトに変換
                    if isinstance(chapter_result, dict):
                        chapter = Chapter(**chapter_result)
                    else:
                        chapter = chapter_result
                        
                    # 章を保存
                    self.state.chapters.append(chapter)
                    
                    # 進捗を更新
                    progress.update(task_id, advance=1)
                    
                    logger.info(f"Chapter {i+1}/{len(self.state.outline.chapter_titles)} generated")
                except Exception as e:
                    logger.error(f"Error generating chapter {i+1} '{title}': {str(e)}")
                    console.print(f"[bold red]Error generating chapter {i+1} '{title}': {str(e)}[/bold red]")
                    # エラーが発生しても続行
                    progress.update(task_id, advance=1)
        
        if CONFIG["image_generation"]:
            return self.generate_image_descriptions()
        else:
            return self.edit_book()
    
    @retry_on_error()
    async def generate_image_descriptions(self):
        """章ごとに画像の説明を生成"""
        with console.status("[bold blue]Generating image descriptions...[/bold blue]"):
            try:
                llm = initialize_llm()
                image_designer_agent = create_image_designer_agent(llm)
                
                for i, chapter in enumerate(self.state.chapters):
                    image_task = create_image_description_task(image_designer_agent, chapter)
                    
                    # 画像説明生成を実行
                    image_crew = Crew(
                        agents=[image_designer_agent],
                        tasks=[image_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    image_descriptions = await image_crew.kickoff(
                        inputs={"chapter": chapter.dict()}
                    )
                    
                    # 画像説明を章に追加
                    if isinstance(image_descriptions, list):
                        self.state.chapters[i].image_descriptions = image_descriptions
                    else:
                        self.state.chapters[i].image_descriptions = [image_descriptions]
                
                logger.info("Image descriptions generated")
                console.print("[bold green]✓ Image descriptions generated[/bold green]")
                
                # 次のステップへ
                return self.edit_book()
            except Exception as e:
                logger.error(f"Image description generation failed: {str(e)}")
                console.print(f"[bold red]Image description generation failed: {str(e)}[/bold red]")
                # エラーが発生しても続行
                return self.edit_book()
    
    @retry_on_error()
    async def edit_book(self):
        """本全体を編集・校正"""
        with console.status("[bold blue]Editing and proofreading book...[/bold blue]"):
            try:
                # 本全体を構成
                book = Book(
                    metadata=self.state.metadata,
                    outline=self.state.outline,
                    chapters=self.state.chapters,
                    references=[]
                )
                
                # 参考文献を集約
                all_references = []
                for chapter in self.state.chapters:
                    for section in chapter.sections:
                        all_references.extend(section.references)
                
                # 重複を除去
                unique_refs = {}
                for ref in all_references:
                    key = ref.source + (ref.url or "")
                    unique_refs[key] = ref
                
                book.references = list(unique_refs.values())
                
                # 編集エージェントを作成
                llm = initialize_llm()
                editor_agent = create_editor_agent(llm)
                editing_task = create_editing_task(editor_agent, book)
                
                # 編集を実行
                editing_crew = Crew(
                    agents=[editor_agent],
                    tasks=[editing_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                edited_book = await editing_crew.kickoff(inputs={"book": book.dict()})
                
                # 結果が辞書なら、Bookオブジェクトに変換
                if isinstance(edited_book, dict):
                    self.state.final_book = Book(**edited_book)
                else:
                    self.state.final_book = edited_book
                
                logger.info("Book editing completed")
                console.print("[bold green]✓ Book editing completed[/bold green]")
                
                # 次のステップへ
                return self.save_book()
            except Exception as e:
                logger.error(f"Book editing failed: {str(e)}")
                console.print(f"[bold red]Book editing failed: {str(e)}[/bold red]")
                # エラーが発生しても続行するため、未編集の本を使用
                book = Book(
                    metadata=self.state.metadata,
                    outline=self.state.outline,
                    chapters=self.state.chapters,
                    references=[]
                )
                self.state.final_book = book
                return self.save_book()
    
    @listen(generate_chapters, edit_book)
    async def save_book(self):
        """本を指定された形式で保存"""
        if not self.state.chapters:
            raise ValueError("No chapters generated")
        
        book = self.state.final_book or Book(
            metadata=self.state.metadata,
            outline=self.state.outline,
            chapters=self.state.chapters,
            references=[]
        )
        
        # 出力ディレクトリの作成
        os.makedirs("output", exist_ok=True)
        
        # ファイル名の安全化
        safe_title = "".join(c if c.isalnum() or c == ' ' else '_' for c in book.metadata.title)
        safe_title = safe_title.replace(' ', '_').lower()
        date_str = datetime.now().strftime("%Y%m%d")
        base_filename = f"output/{safe_title}_{date_str}"
        
        # MDでの保存
        if self.state.output_format == "md" or self.state.output_format == "all":
            md_filename = f"{base_filename}.md"
            self._save_markdown(book, md_filename)
            console.print(f"[bold green]✓ Book saved as Markdown: {md_filename}[/bold green]")
        
        # HTMLでの保存
        if self.state.output_format == "html" or self.state.output_format == "all":
            html_filename = f"{base_filename}.html"
            self._save_html(book, html_filename)
            console.print(f"[bold green]✓ Book saved as HTML: {html_filename}[/bold green]")
        
        # PDFでの保存
        if self.state.output_format == "pdf" or self.state.output_format == "all":
            try:
                import weasyprint
                html_filename = f"{base_filename}_temp.html"
                pdf_filename = f"{base_filename}.pdf"
                self._save_html(book, html_filename)
                weasyprint.HTML(filename=html_filename).write_pdf(pdf_filename)
                os.remove(html_filename)  # 一時HTMLファイルを削除
                console.print(f"[bold green]✓ Book saved as PDF: {pdf_filename}[/bold green]")
            except ImportError:
                logger.warning("WeasyPrint not installed. PDF export not available.")
                console.print("[bold yellow]⚠ WeasyPrint not installed. PDF export not available.[/bold yellow]")
        
        logger.info(f"Book saved in format: {self.state.output_format}")
        console.print("\n[bold green]Book creation completed! 📚[/bold green]")
        return book
    
    def _save_markdown(self, book, filename):
        """本をMarkdownとして保存"""
        with open(filename, "w", encoding="utf-8") as f:
            # タイトルとメタデータ
            f.write(f"# {book.metadata.title}\n\n")
            if book.metadata.subtitle:
                f.write(f"## {book.metadata.subtitle}\n\n")
            f.write(f"By: {book.metadata.author}\n\n")
            f.write(f"Date: {book.metadata.date_created}\n\n")
            
            # 目次
            f.write("## Table of Contents\n\n")
            for i, chapter in enumerate(book.chapters):
                f.write(f"{i+1}. [{chapter.title}](#chapter-{i+1})\n")
            f.write("\n\n")
            
            # 各章
            for i, chapter in enumerate(book.chapters):
                f.write(f"# Chapter {i+1}: {chapter.title}\n\n")
                
                # 章の概要
                if chapter.summary:
                    f.write(f"*{chapter.summary}*\n\n")
                
                # 章のセクション
                for section in chapter.sections:
                    if section.type == "paragraph":
                        f.write(f"{section.content}\n\n")
                    elif section.type == "list_item":
                        f.write(f"- {section.content}\n")
                    elif section.type == "code":
                        f.write(f"```\n{section.content}\n```\n\n")
                    elif section.type == "quote":
                        f.write(f"> {section.content}\n\n")
                    else:
                        f.write(f"{section.content}\n\n")
                
                # 画像の説明
                if chapter.image_descriptions:
                    f.write("\n### Chapter Images\n\n")
                    for i, desc in enumerate(chapter.image_descriptions):
                        f.write(f"*Image {i+1}: {desc}*\n\n")
                
                f.write("\n\n")
            
            # 参考文献
            if book.references:
                f.write("## References\n\n")
                for i, ref in enumerate(book.references):
                    f.write(f"{i+1}. {ref.source}")
                    if ref.url:
                        f.write(f" [{ref.url}]")
                    f.write(f" (Accessed: {ref.accessed_date})\n")
    
    def _save_html(self, book, filename):
        """本をHTMLとして保存"""
        env = Environment(loader=FileSystemLoader("."))
        
        # テンプレートが存在しない場合は作成
        template_str = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ book.metadata.title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        .chapter {
            margin-bottom: 40px;
        }
        .summary {
            font-style: italic;
            color: #555;
            margin-bottom: 20px;
        }
        .toc {
            background-color: #f8f8f8;
            padding: 20px;
            border-radius: 5px;
        }
        .references {
            border-top: 1px solid #ddd;
            padding-top: 20px;
            margin-top: 40px;
        }
        .image-description {
            background-color: #f0f0f0;
            padding: 10px;
            border-left: 4px solid #ccc;
            font-style: italic;
        }
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            font-family: monospace;
        }
        pre {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        blockquote {
            border-left: 4px solid #ccc;
            padding-left: 15px;
            color: #555;
        }
    </style>
</head>
<body>
    <header>
        <h1>{{ book.metadata.title }}</h1>
        {% if book.metadata.subtitle %}
        <h2>{{ book.metadata.subtitle }}</h2>
        {% endif %}
        <p>By: {{ book.metadata.author }}</p>
        <p>Date: {{ book.metadata.date_created }}</p>
    </header>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ol>
            {% for chapter in book.chapters %}
            <li><a href="#chapter-{{ loop.index }}">{{ chapter.title }}</a></li>
            {% endfor %}
        </ol>
    </div>
    
    {% for chapter in book.chapters %}
    <div class="chapter" id="chapter-{{ loop.index }}">
        <h2>Chapter {{ loop.index }}: {{ chapter.title }}</h2>
        
        {% if chapter.summary %}
        <div class="summary">{{ chapter.summary }}</div>
        {% endif %}
        
        {% for section in chapter.sections %}
            {% if section.type == "paragraph" %}
                <p>{{ section.content }}</p>
            {% elif section.type == "list_item" %}
                <ul><li>{{ section.content }}</li></ul>
            {% elif section.type == "code" %}
                <pre><code>{{ section.content }}</code></pre>
            {% elif section.type == "quote" %}
                <blockquote>{{ section.content }}</blockquote>
            {% else %}
                <p>{{ section.content }}</p>
            {% endif %}
        {% endfor %}
        
        {% if chapter.image_descriptions %}
        <div class="image-section">
            <h3>Chapter Images</h3>
            {% for desc in chapter.image_descriptions %}
            <div class="image-description">Image {{ loop.index }}: {{ desc }}</div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    {% endfor %}
    
    {% if book.references %}
    <div class="references">
        <h2>References</h2>
        <ol>
            {% for ref in book.references %}
            <li>
                {{ ref.source }}
                {% if ref.url %}
                <a href="{{ ref.url }}">[Link]</a>
                {% endif %}
                (Accessed: {{ ref.accessed_date }})
            </li>
            {% endfor %}
        </ol>
    </div>
    {% endif %}
</body>
</html>"""
        
        with open("book_template.html", "w", encoding="utf-8") as f:
            f.write(template_str)
        
        template = env.get_template("book_template.html")
        html_content = template.render(book=book)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # テンプレートファイルを削除
        os.remove("book_template.html")

# メイン処理
async def main_async():
    """メイン関数（非同期版）"""
    # 引数の解析
    parser = argparse.ArgumentParser(description="AI-powered book writer")
    parser.add_argument("--topic", "-t", type=str, required=True, help="Topic of the book to write")
    parser.add_argument("--format", "-f", type=str, choices=CONFIG["output_formats"] + ["all"], 
                        default=CONFIG["default_output_format"], help="Output format")
    parser.add_argument("--model", "-m", type=str, help="LLM model to use")
    parser.add_argument("--images", "-i", action="store_true", help="Generate image descriptions")
    parser.add_argument("--verify", "-v", action="store_true", help="Verify facts")
    
    args = parser.parse_args()
    
    # 設定の更新
    if args.model:
        CONFIG["model"] = args.model
    CONFIG["image_generation"] = args.images
    CONFIG["verify_facts"] = args.verify
    
    # コンソールへの出力
    console.print(f"[bold magenta]📚 AI Book Writer[/bold magenta]")
    console.print(f"Topic: {args.topic}")
    console.print(f"Model: {CONFIG['model']}")
    console.print(f"Output format: {args.format}")
    console.print(f"Generate images: {'Yes' if CONFIG['image_generation'] else 'No'}")
    console.print(f"Verify facts: {'Yes' if CONFIG['verify_facts'] else 'No'}")
    console.print("")
    
    # 初期状態の作成
    initial_state = BookState(
        topic=args.topic,
        output_format=args.format
    )
    
    # フローの実行
    book_flow = BookFlow(state=initial_state)
    final_book = await book_flow.start()
    
    return final_book

def main():
    """メイン関数（同期版）"""
    # Windows向けの非同期実行対応
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 