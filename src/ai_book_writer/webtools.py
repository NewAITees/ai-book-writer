#!/usr/bin/env python3
"""
Web scraping and search tools using Crawl4AI.
This module provides a clean interface for retrieving information from the web.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Set, Pattern
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, validator, HttpUrl, AnyHttpUrl, ValidationError, field_validator, ValidationInfo
from pydantic_settings import BaseSettings


from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import aiohttp
import chardet
import yarl
import requests
from concurrent.futures import ThreadPoolExecutor
import tldextract
from urllib3.util import parse_url
import socket
import ipaddress

# Configure logging
logger = logging.getLogger(__name__)

class WebToolsConfig(BaseSettings):
    """Configuration for web tools"""
    search_provider: str = "duckduckgo"
    max_results: int = 10
    default_timeout: int = 30
    user_agent: str = "Mozilla/5.0 (compatible; AIBookWriter/1.0)"
    crawl_delay: float = 1.0  # Seconds between requests
    timeout: int = 30
    max_retries: int = 3
    delay_between_requests: float = 1.0
    
    @field_validator('user_agent')
    def sanitize_user_agent(cls, v: str) -> str:
        """Remove potentially dangerous characters from user agent"""
        return re.sub(r'[\r\n\t]', '', v)
    
    class Config:
        env_prefix = "WEBTOOLS_"

# Global configuration instance
config = WebToolsConfig()

class WebContentExtractionError(Exception):
    """Webコンテンツ抽出時のエラーを表す例外クラス"""
    pass

class Link(BaseModel):
    """リンクを表すクラス"""
    url: str
    text: Optional[str] = None
    domain: Optional[str] = None

    @validator('url')
    def validate_and_normalize_url(cls, v: str) -> str:
        """URLを検証して正規化する"""
        try:
            parsed = urlparse(v)
            
            # 許可されたスキームをチェック
            allowed_schemes = ['http', 'https', 'mailto', 'javascript']
            if parsed.scheme and parsed.scheme.lower() not in allowed_schemes:
                raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
            
            # パストラバーサルをチェック
            if '..' in parsed.path or '/etc/' in parsed.path:
                raise ValueError(f"Path traversal detected in URL: {v}")
            
            # プライベートIPアドレスをチェック（httpとhttpsのみ）
            if parsed.scheme in ['http', 'https'] and parsed.netloc:
                host = parsed.netloc.split(':')[0]
                try:
                    ip = ipaddress.ip_address(socket.gethostbyname(host))
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        raise ValueError(f"Private IP address detected in URL: {v}")
                except (socket.gaierror, ValueError):
                    pass
            
            return v
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid URL: {v}")

    @validator('domain', pre=True, always=True)
    def extract_domain_from_url(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        if v is not None:
            return v
        
        url = values.get('url')
        if not url:
            return None

        try:
            # javascriptスキームの場合はNoneを返す
            if url.startswith('javascript:'):
                return None

            # mailtoスキームの場合はNoneを返す
            if url.startswith('mailto:'):
                return None

            # 相対パスの場合はNoneを返す
            if not urlparse(url).netloc:
                return None

            # プロトコル相対URLの場合はドメインを抽出
            if url.startswith('//'):
                url = f'https:{url}'

            # ドメインを抽出
            extracted = tldextract.extract(url)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}".lower()
            
            return None
        except Exception:
            return None

class WebPageContent(BaseModel):
    """Represents extracted webpage content"""
    url: str
    base_url: Optional[str] = None
    title: str = ''
    content: str = ''
    html: str
    links: List[Link] = []
    internal_links: List[Link] = []
    external_links: List[Link] = []
    images: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator('base_url', mode='before')
    def set_base_url(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Set base URL from the full URL if not provided"""
        if v is not None:
            return v
        
        url = info.data.get('url')
        if not url:
            return None
            
        try:
            return str(yarl.URL(url).origin())
        except Exception:
            return None
    
    def resolve_relative_links(self):
        """Resolve all relative links using the base URL"""
        resolved_links = []
        
        # Get the base domain for determining internal vs external
        base_domain = urlparse(self.base_url).netloc
        internal_links = []
        external_links = []
        
        for link in self.links:
            url = link.url
            # Handle relative URLs
            if url.startswith('/'):
                url = urljoin(self.base_url, url)
                link_is_external = False
            elif not url.startswith(('http://', 'https://')):
                url = urljoin(self.base_url, '/' + url)
                link_is_external = False
            else:
                # For absolute URLs, check if they're on the same domain
                link_domain = urlparse(url).netloc
                link_is_external = link_domain != base_domain
            
            resolved_link = Link(
                url=url,
                domain=urlparse(url).netloc
            )
            
            resolved_links.append(resolved_link)
            
            if link_is_external:
                external_links.append(resolved_link)
            else:
                internal_links.append(resolved_link)
        
        # Update the links lists
        self.links = resolved_links
        self.internal_links = internal_links
        self.external_links = external_links
        
        return self

class SearchResult(BaseModel):
    """Search result from a web search"""
    title: str
    snippet: str = ""
    url: str
    
    @validator('url')
    def validate_url(cls, v):
        """Ensure URL is properly formatted"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class SearchResponse(BaseModel):
    """Complete search response with results and metadata"""
    query: str
    results: List[SearchResult]
    timestamp: datetime = Field(default_factory=datetime.now)
    total_results_found: int = 0
    success: bool = True
    error_message: Optional[str] = None

async def search_duckduckgo(query: str, max_results: int = 10) -> SearchResponse:
    """
    Search using DuckDuckGo and return structured results
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        SearchResponse: Structured search results
    """
    try:
        # DDGS is synchronous, so run in executor
        loop = asyncio.get_running_loop()
        raw_results = await loop.run_in_executor(
            None, 
            lambda: list(DDGS().text(
                keywords=query,
                region='jp-jp',
                safesearch='off',
                timelimit=None,
                max_results=max_results
            ))
        )
        
        # Transform results to our model
        search_results = []
        for result in raw_results:
            try:
                search_results.append(
                    SearchResult(
                        title=result.get("title", "No title"),
                        snippet=result.get("body", ""),
                        url=result.get("href", "")
                    )
                )
            except ValueError as e:
                logger.warning(f"Skipping invalid search result: {e}")
                continue
        
        return SearchResponse(
            query=query,
            results=search_results,
            total_results_found=len(search_results),
            success=True
        )
        
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {str(e)}")
        return SearchResponse(
            query=query,
            results=[],
            success=False,
            error_message=str(e)
        )

def sanitize_html_content(soup: BeautifulSoup) -> str:
    """
    Sanitize HTML content by removing potentially dangerous elements and attributes.
    
    Args:
        soup: BeautifulSoup object containing the HTML content
        
    Returns:
        Sanitized content as string
    """
    # Remove script, style, and other dangerous tags
    for tag in soup.find_all(['script', 'style', 'iframe', 'object', 'embed', 'noscript', 'meta', 'link']):
        tag.decompose()
    
    # Remove on* attributes and dangerous URLs
    dangerous_attrs = {'onload', 'onerror', 'onclick', 'onmouseover', 'onsubmit', 'onkeyup', 'onkeydown'}
    dangerous_schemes = {'javascript:', 'data:', 'file:', 'vbscript:'}
    
    for tag in soup.find_all(True):
        # Remove dangerous attributes
        for attr in list(tag.attrs):
            if attr.lower().startswith('on') or attr.lower() in dangerous_attrs:
                del tag.attrs[attr]
            elif attr in ['href', 'src']:
                value = tag.attrs[attr].lower()
                if any(scheme in value for scheme in dangerous_schemes):
                    del tag.attrs[attr]
    
    # Extract text content from safe elements
    content = []
    safe_tags = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'span', 'article', 'section', 'main'}
    
    for tag in soup.find_all(safe_tags):
        text = tag.get_text(strip=True)
        if text:
            content.append(text)
    
    return '\n'.join(content) if content else "No content available"

def force_utf8_encoding(content: str) -> str:
    """
    Force content to UTF-8 encoding.
    
    Args:
        content: Content to encode
        
    Returns:
        UTF-8 encoded content
    """
    try:
        # Try to detect the encoding
        detected = chardet.detect(content.encode())
        if detected and detected['encoding']:
            # Decode with detected encoding and re-encode as UTF-8
            content = content.encode(detected['encoding']).decode('utf-8', errors='replace')
    except Exception:
        # If any error occurs, force UTF-8 with replacement
        content = content.encode('utf-8', errors='replace').decode('utf-8')
    return content

def is_dangerous_url(url: str) -> bool:
    """
    URLが危険かどうかを判定する
    """
    parsed = urlparse(url)
    
    # 許可されたスキームをチェック
    allowed_schemes = ['http', 'https', 'mailto', 'javascript']
    if parsed.scheme and parsed.scheme.lower() not in allowed_schemes:
        return True
    
    # パストラバーサルをチェック
    if '..' in parsed.path or '/etc/' in parsed.path:
        return True
    
    # プライベートIPアドレスをチェック
    if parsed.scheme in ['http', 'https'] and parsed.netloc:
        host = parsed.netloc.split(':')[0]
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return True
        except (socket.gaierror, ValueError):
            pass
    
    return False

def validate_url(url: str) -> bool:
    """
    URLが有効かどうかを検証する
    """
    try:
        parsed = urlparse(url)
        
        # スキームのチェック
        if not parsed.scheme and not parsed.netloc and not parsed.path.startswith('/'):
            return False
        
        # 危険なURLをチェック
        if is_dangerous_url(url):
            raise ValueError(f"Dangerous URL detected: {url}")
        
        # ドメインの解決をチェック（httpとhttpsのみ）
        if parsed.scheme in ['http', 'https'] and parsed.netloc:
            try:
                socket.gethostbyname(parsed.netloc)
            except socket.gaierror:
                return False
        
        return True
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        return False

async def extract_webpage_content(url: str) -> WebPageContent:
    """
    指定されたURLからコンテンツを抽出する
    """
    try:
        # URLの検証
        if not validate_url(url):
            return WebPageContent(
                url=url,
                base_url=url,
                title="Error: Invalid URL",
                content=f"Failed to extract content: Invalid URL or domain ({url})",
                html="",
                links=[]
            )
        
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        # エンコーディングを検出して適切に処理
        raw_content = response.content
        detected = chardet.detect(raw_content)
        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
        try:
            content = raw_content.decode(encoding)
            # UTF-8に変換
            if encoding.lower() not in ['utf-8', 'ascii']:
                content = content.encode('utf-8', errors='ignore').decode('utf-8')
        except UnicodeDecodeError:
            content = raw_content.decode('utf-8', errors='ignore')
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # スクリプトタグと危険な要素を削除
        for script in soup.find_all(['script', 'style', 'iframe', 'noscript', 'meta', 'link']):
            script.decompose()
        
        # リンクを抽出
        links = []
        for a in soup.find_all('a', href=True):
            href = a.get('href', '').strip()
            if href:
                try:
                    # 相対URLを絶対URLに変換
                    if not urlparse(href).netloc:
                        href = urljoin(url, href)
                    
                    link = Link(url=href, text=a.get_text(strip=True))
                    links.append(link)
                except Exception as e:
                    logging.warning(f"Failed to process link {href}: {str(e)}")
        
        # タイトルを抽出
        title = soup.title.string if soup.title else "Untitled Page"
        
        # コンテンツを抽出
        content = ' '.join(soup.stripped_strings)
        
        return WebPageContent(
            url=url,
            base_url=url,
            title=title,
            content=content,
            html=str(soup),
            links=links
        )
    except requests.exceptions.RequestException as e:
        return WebPageContent(
            url=url,
            base_url=url,
            title="Error: Request Failed",
            content=f"Failed to extract content: {str(e)}",
            html="",
            links=[]
        )
    except ValueError as e:
        raise  # 危険なURLの場合はValueErrorをそのまま再送出
    except Exception as e:
        return WebPageContent(
            url=url,
            base_url=url,
            title="Error: Extraction Failed",
            content=f"Failed to extract content: {str(e)}",
            html="",
            links=[]
        )

async def search_and_extract(query: str, max_results: int = 5) -> List[WebPageContent]:
    """
    Search for information and extract content from the top results
    
    Args:
        query: Search query
        max_results: Maximum number of results to process
        
    Returns:
        List[WebPageContent]: Content extracted from the top search results
    """
    # Search for information
    search_response = await search_duckduckgo(query, max_results * 2)  # Get extra in case some fail
    
    if not search_response.success or not search_response.results:
        logger.warning(f"Search for '{query}' returned no results")
        return []
    
    # Extract content from top results
    content_list = []
    tasks = []
    
    # Create extraction tasks
    for result in search_response.results[:max_results]:
        tasks.append(extract_webpage_content(result.url))
    
    # Run extractions concurrently
    if tasks:
        extracted_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful extractions
        for content in extracted_contents:
            if isinstance(content, WebPageContent) and content.content:
                content_list.append(content)
    
    logger.info(f"Extracted content from {len(content_list)} pages for query '{query}'")
    return content_list

def fetch_webpage_content_sync(url: str) -> WebPageContent:
    """
    Synchronous wrapper for webpage content extraction
    
    Args:
        url: URL to extract content from
        
    Returns:
        WebPageContent: Structured content from the webpage
    """
    return asyncio.run(extract_webpage_content(url))

def filter_links(links: List[Link], include_domains: Optional[List[str]] = None, 
                exclude_domains: Optional[List[str]] = None,
                patterns: Optional[List[str]] = None) -> List[Link]:
    """
    指定されたドメインとパターンに基づいてリンクをフィルタリングする
    """
    filtered_links = []
    
    for link in links:
        if not link.domain:
            continue
            
        domain = link.domain.lower()
        
        # 除外ドメインのチェック
        if exclude_domains and any(d.lower() == domain for d in exclude_domains):
            continue
            
        # 含めるドメインのチェック
        if include_domains:
            if not any(d.lower() == domain for d in include_domains):
                continue
        
        # パターンのチェック
        if patterns:
            if not any(re.search(pattern, link.url) for pattern in patterns):
                continue
            
        filtered_links.append(link)
    
    return filtered_links

def extract_main_content_sections(html: str) -> Dict[str, str]:
    """
    Extract main content sections from HTML
    
    Args:
        html: HTML content
        
    Returns:
        Dict[str, str]: Dictionary of section name to content
    """
    sections = {}
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try to find main content area
        main_content = soup.find('main') or soup.find(id='content') or soup.find(id='main')
        
        if not main_content:
            # Fall back to article or body
            main_content = soup.find('article') or soup.find('body')
        
        if main_content:
            # Extract sections by headers
            current_section = "introduction"
            current_content = []
            
            for elem in main_content.children:
                if elem.name in ('h1', 'h2', 'h3'):
                    # Save previous section
                    if current_content:
                        sections[current_section] = "\n".join(current_content)
                    
                    # Start new section
                    current_section = elem.get_text(strip=True)
                    current_content = []
                elif elem.name:
                    # Add to current section
                    current_content.append(elem.get_text(strip=True))
            
            # Save last section
            if current_content:
                sections[current_section] = "\n".join(current_content)
    
    except Exception as e:
        logger.error(f"Failed to extract content sections: {str(e)}")
    
    return sections

async def crawl_links(base_content: WebPageContent, max_depth: int = 1, 
                     internal_only: bool = True, max_pages: int = 10) -> Dict[str, WebPageContent]:
    """
    Crawl links found on a webpage
    
    Args:
        base_content: WebPageContent to start crawling from
        max_depth: Maximum crawl depth
        internal_only: Whether to only crawl internal links
        max_pages: Maximum number of pages to crawl
        
    Returns:
        Dict[str, WebPageContent]: Dictionary of URL to content for all crawled pages
    """
    results = {base_content.url: base_content}
    to_visit = []
    
    # Initialize links to visit
    links_to_add = base_content.internal_links if internal_only else base_content.links
    for link in links_to_add:
        to_visit.append((link.url, 1))  # (url, depth)
    
    # Set to track visited URLs
    visited = {base_content.url}
    
    # Crawl links
    while to_visit and len(results) < max_pages:
        url, depth = to_visit.pop(0)
        
        # Skip if already visited or exceeds max depth
        if url in visited or depth > max_depth:
            continue
        
        visited.add(url)
        
        # Add delay between requests
        await asyncio.sleep(config.crawl_delay)
        
        # Crawl page
        content = await extract_webpage_content(url)
        results[url] = content
        
        # Add links to visit if not at max depth
        if depth < max_depth:
            links_to_add = content.internal_links if internal_only else content.links
            for link in links_to_add:
                if link.url not in visited:
                    to_visit.append((link.url, depth + 1))
    
    return results

async def search_extract_and_crawl(query: str, max_results: int = 3, 
                                 crawl_depth: int = 1, max_pages_per_result: int = 5) -> Dict[str, Dict[str, WebPageContent]]:
    """
    Search for information, extract content from top results, and crawl their links
    
    Args:
        query: Search query
        max_results: Maximum number of search results to process
        crawl_depth: Maximum depth for crawling
        max_pages_per_result: Maximum pages to crawl per search result
        
    Returns:
        Dict[str, Dict[str, WebPageContent]]: Nested dictionary of search result URLs to crawled content
    """
    # Search for information
    search_response = await search_duckduckgo(query, max_results * 2)
    
    if not search_response.success or not search_response.results:
        logger.warning(f"Search for '{query}' returned no results")
        return {}
    
    # Extract content from top results and crawl links
    crawl_results = {}
    
    for result in search_response.results[:max_results]:
        # First extract content from the search result
        base_content = await extract_webpage_content(result.url)
        
        # Then crawl links from this page
        crawled_pages = await crawl_links(
            base_content, 
            max_depth=crawl_depth, 
            internal_only=True,
            max_pages=max_pages_per_result
        )
        
        # Add to results
        crawl_results[result.url] = crawled_pages
    
    return crawl_results

# Export public API
__all__ = [
    'SearchResult', 
    'WebPageContent', 
    'SearchResponse',
    'WebToolsConfig',
    'Link',
    'config',
    'search_duckduckgo',
    'extract_webpage_content',
    'search_and_extract',
    'search_extract_and_crawl',
    'crawl_links',
    'filter_links',
    'extract_main_content_sections',
    'fetch_webpage_content_sync',
    'validate_url'
]

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def run_tests():
        """Run various tests of the web scraping functionality"""
        
        # Test 1: Basic search and content extraction
        print("\n=== Test 1: Basic Search and Content Extraction ===")
        query = "Python programming guide"
        results = await search_and_extract(query, max_results=2)
        
        for result in results:
            print(f"\nTitle: {result.title}")
            print(f"URL: {result.url}")
            print(f"Internal links: {len(result.internal_links)}")
            print(f"External links: {len(result.external_links)}")
            print("---")
        
        # Test 2: Extract content and crawl internal links
        if results:
            print("\n=== Test 2: Link Crawling ===")
            first_result = results[0]
            crawled_pages = await crawl_links(
                first_result,
                max_depth=1,
                internal_only=True,
                max_pages=3
            )
            
            print(f"\nCrawled {len(crawled_pages)} pages from {first_result.url}")
            
            # Print titles of crawled pages
            for url, content in crawled_pages.items():
                print(f"\n  Page: {content.title}")
                print(f"  URL: {url}")
                print(f"  Internal links: {len(content.internal_links)}")
                print(f"  External links: {len(content.external_links)}")
        
        # Test 3: Full search, extract and crawl
        print("\n=== Test 3: Search, Extract and Crawl ===")
        crawl_results = await search_extract_and_crawl(
            "Machine learning tutorial",
            max_results=2,
            crawl_depth=1,
            max_pages_per_result=3
        )
        
        for base_url, pages in crawl_results.items():
            print(f"\nFrom {base_url}, crawled {len(pages)} pages")
            base_page = pages[base_url]
            
            # Show the first 3 internal links from the base page
            print("\nInternal links found:")
            for i, link in enumerate(base_page.internal_links[:3], 1):
                print(f"  {i}. {link.text or link.url}")
        
        # Test 4: Content section extraction
        if results:
            print("\n=== Test 4: Content Section Extraction ===")
            first_page = results[0]
            if first_page.html:
                sections = extract_main_content_sections(first_page.html)
                print(f"\nExtracted {len(sections)} sections from {first_page.url}")
                for section_name, content in sections.items():
                    print(f"\nSection: {section_name}")
                    print(f"Content length: {len(content)} characters")
                    print(f"Preview: {content[:100]}...")
        
        # Test 5: Link filtering
        if results:
            print("\n=== Test 5: Link Filtering ===")
            first_page = results[0]
            
            # Filter links to only include those with 'python' in the URL
            python_links = filter_links(
                first_page.links,
                include_patterns=[r'python'],
                exclude_domains=['google.com', 'youtube.com']
            )
            
            print(f"\nFiltered links containing 'python':")
            for i, link in enumerate(python_links[:5], 1):
                print(f"  {i}. {link.text or link.url}")
    
    # Run all tests
    asyncio.run(run_tests()) 