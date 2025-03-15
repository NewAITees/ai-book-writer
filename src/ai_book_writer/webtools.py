#!/usr/bin/env python3
"""
Web scraping and search tools using Crawl4AI.
This module provides a clean interface for retrieving information from the web.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

# ロギング設定
logger = logging.getLogger("webtools")

class SearchResult(BaseModel):
    """検索結果の構造"""
    title: str
    snippet: str
    link: str

class WebContent(BaseModel):
    """Webページの内容"""
    url: str
    title: str
    content: str
    markdown: str
    html: Optional[str] = None
    links: List[str] = []
    images: List[str] = []

async def fetch_webpage_content_async(url: str, **kwargs) -> WebContent:
    """
    指定されたURLからウェブページの内容を非同期で取得
    
    Args:
        url: コンテンツを取得するURL
        **kwargs: Crawl4AIの設定パラメータ
        
    Returns:
        WebContent: ウェブページの内容
    """
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, **kwargs)
            
            return WebContent(
                url=url,
                title=result.metadata.get("title", ""),
                content=result.text,
                markdown=result.markdown,
                html=result.html,
                links=result.links,
                images=result.images
            )
    except Exception as e:
        logger.error(f"Error fetching webpage {url}: {str(e)}")
        # 基本的な情報だけでも返す
        return WebContent(
            url=url,
            title="Error fetching page",
            content=f"Error: {str(e)}",
            markdown=f"# Error\n\nFailed to fetch content from {url}: {str(e)}",
            links=[],
            images=[]
        )

def fetch_webpage_content(url: str, **kwargs) -> WebContent:
    """
    指定されたURLからウェブページの内容を取得（同期版）
    
    Args:
        url: コンテンツを取得するURL
        **kwargs: Crawl4AIの設定パラメータ
        
    Returns:
        WebContent: ウェブページの内容
    """
    return asyncio.run(fetch_webpage_content_async(url, **kwargs))

async def search_and_extract_content(query: str, num_results: int = 5) -> List[WebContent]:
    """
    検索クエリを実行し、検索結果から内容を抽出
    
    Args:
        query: 検索クエリ
        num_results: 結果の最大数
        
    Returns:
        List[WebContent]: ウェブページの内容のリスト
    """
    try:
        # 最初に検索結果を取得
        search_url = f"https://news.google.com/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
        
        browser_config = BrowserConfig(
            headless=True,
            timeout=30000  # 30秒
        )
        
        run_config = CrawlerRunConfig(
            max_content_length=50000,  # 最大コンテンツ長（文字数）
            extract_links=True
        )
        
        async with AsyncWebCrawler() as crawler:
            search_result = await crawler.arun(
                url=search_url,
                browser_config=browser_config,
                run_config=run_config
            )
        
        # 検索結果から最初のnum_results個のリンクを取得
        content_links = search_result.links[:num_results]
        
        # 各リンクの内容を非同期で取得
        tasks = []
        for link in content_links:
            tasks.append(fetch_webpage_content_async(link))
        
        # 非同期で全てのリンクからコンテンツを取得
        contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 例外をフィルタリング
        valid_contents = []
        for content in contents:
            if not isinstance(content, Exception):
                valid_contents.append(content)
        
        return valid_contents
    
    except Exception as e:
        logger.error(f"Error in search and extract: {str(e)}")
        return []

# メインの部分（テスト用）
if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # クローラーのテスト
    url = "https://www.python.org/"
    print(f"Fetching content from: {url}")
    
    content = fetch_webpage_content(url)
    
    print(f"Title: {content.title}")
    print(f"Content length: {len(content.content)} characters")
    print(f"Markdown length: {len(content.markdown)} characters")
    print(f"Sample markdown: {content.markdown[:300]}...")
    print(f"Found {len(content.links)} links and {len(content.images)} images")
    
    # 検索と内容抽出のテスト
    async def test_search():
        query = "Python programming language"
        print(f"\nSearching for: {query}")
        
        results = await search_and_extract_content(query, 3)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.title}")
            print(f"   Content length: {len(result.content)} characters")
            print(f"   URL: {result.url}\n")
    
    asyncio.run(test_search()) 