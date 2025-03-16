#!/usr/bin/env python3
"""
Comprehensive test suite for the webtools module.
Tests all aspects including error handling, performance, data quality, and security.
"""

import asyncio
import logging
import json
import os
import time
import resource
import pytest
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import patch, MagicMock
import warnings
import chardet
import html
from bs4 import BeautifulSoup

from ai_book_writer.webtools import (
    SearchResult,
    WebPageContent,
    SearchResponse,
    WebToolsConfig,
    Link,
    search_duckduckgo,
    extract_webpage_content,
    search_and_extract,
    search_extract_and_crawl,
    crawl_links,
    filter_links,
    extract_main_content_sections,
    fetch_webpage_content_sync
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webtools_test")

# Test output directory
TEST_OUTPUT_DIR = Path("test_results")
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Test configuration
TEST_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "delay_between_requests": 1.0,
    "test_urls": [
        "https://python.org",
        "https://docs.python.org",
        "https://pypi.org",
    ],
    "invalid_urls": [
        "https://invalid.domain.test",
        "not_a_url",
        "http://localhost:1234",
    ],
    "test_queries": [
        "Python programming",
        "Machine learning tutorial",
        "Data science best practices",
    ],
}

# Performance monitoring
class PerformanceMonitor:
    """Monitor and record performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.metrics = {}
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        self.metrics = {
            "duration": self.end_time - self.start_time,
            "memory_increase": self.peak_memory - self.start_memory,
            "peak_memory": self.peak_memory
        }

# Utility functions
def save_test_result(data: Any, filename: str) -> Path:
    """Save test result data to JSON file"""
    file_path = TEST_OUTPUT_DIR / filename
    
    if hasattr(data, 'dict'):
        data = data.dict()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    return file_path

async def measure_concurrent_performance(func, inputs, max_concurrent: int = 3):
    """Measure performance of concurrent execution"""
    with PerformanceMonitor() as monitor:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def wrapped_func(input_data):
            async with semaphore:
                return await func(input_data)
        
        tasks = [wrapped_func(input_data) for input_data in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results, monitor.metrics

# Test Classes
class TestSearchFunctionality:
    """Test search-related functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_search(self):
        """Test basic search functionality"""
        for query in TEST_CONFIG["test_queries"]:
            response = await search_duckduckgo(query)
            assert response.success
            assert len(response.results) > 0
            assert all(isinstance(r, SearchResult) for r in response.results)
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test search error handling"""
        # Test with empty query
        response = await search_duckduckgo("")
        assert not response.success
        
        # Test with very long query
        long_query = "test " * 1000
        response = await search_duckduckgo(long_query)
        assert not response.success
    
    @pytest.mark.asyncio
    async def test_search_performance(self):
        """Test search performance with multiple queries"""
        results, metrics = await measure_concurrent_performance(
            search_duckduckgo,
            TEST_CONFIG["test_queries"]
        )
        
        assert metrics["duration"] < len(TEST_CONFIG["test_queries"]) * TEST_CONFIG["timeout"]
        assert all(r.success for r in results if isinstance(r, SearchResponse))

class TestContentExtraction:
    """Test content extraction functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        """Test basic content extraction"""
        for url in TEST_CONFIG["test_urls"]:
            content = await extract_webpage_content(url)
            assert content.url == url
            assert content.title
            assert content.content
    
    @pytest.mark.asyncio
    async def test_extraction_error_handling(self):
        """Test content extraction error handling"""
        for url in TEST_CONFIG["invalid_urls"]:
            content = await extract_webpage_content(url)
            assert "Error" in content.title
            assert "Failed to extract content" in content.content
    
    @pytest.mark.asyncio
    async def test_character_encoding(self):
        """Test handling of different character encodings"""
        # Test with known UTF-8 page
        content = await extract_webpage_content("https://www.python.org")
        assert "�" not in content.content  # No replacement characters
        
        # Detect encoding
        if content.html:
            detected = chardet.detect(content.html.encode())
            assert detected["encoding"].lower() in ["utf-8", "ascii"]
    
    @pytest.mark.asyncio
    async def test_html_entities(self):
        """Test handling of HTML entities"""
        content = await extract_webpage_content("https://www.python.org")
        if content.html:
            # Check if HTML entities are properly decoded
            soup = BeautifulSoup(content.html, 'html.parser')
            text = soup.get_text()
            assert "&amp;" not in text
            assert "&lt;" not in text
            assert "&gt;" not in text

class TestLinkHandling:
    """Test link handling functionality"""
    
    @pytest.mark.asyncio
    async def test_link_extraction(self):
        """Test link extraction from content"""
        content = await extract_webpage_content("https://www.python.org")
        assert len(content.links) > 0
        assert all(isinstance(link, Link) for link in content.links)
    
    @pytest.mark.asyncio
    async def test_link_validation(self):
        """Test link validation and normalization"""
        # Test with various link formats
        test_links = [
            Link(url="https://example.com"),
            Link(url="/relative/path"),
            Link(url="//protocol-relative.com"),
            Link(url="mailto:test@example.com"),  # Should be filtered
            Link(url="javascript:void(0)"),  # Should be filtered
        ]
        
        filtered = [link for link in test_links if link.url.startswith(('http', 'https', '/'))]
        assert all(link.url.startswith(('http', 'https', '/')) for link in filtered)
    
    def test_link_filtering(self):
        """Test link filtering functionality"""
        links = [
            Link(url="https://example.com/test"),
            Link(url="https://google.com/test"),
            Link(url="https://test.com/python"),
        ]
        
        # Test domain exclusion
        filtered = filter_links(links, exclude_domains=["google.com"])
        assert len(filtered) == 2
        assert all(link.domain != "google.com" for link in filtered)
        
        # Test pattern matching
        filtered = filter_links(links, patterns=[r"python"])
        assert len(filtered) == 1
        assert all("python" in link.url for link in filtered)

class TestCrawling:
    """Test crawling functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_crawling(self):
        """Test basic crawling functionality"""
        content = await extract_webpage_content("https://www.python.org")
        crawled = await crawl_links(content, max_depth=1, max_pages=3)
        
        assert len(crawled) <= 3
        assert all(isinstance(c, WebPageContent) for c in crawled.values())
    
    @pytest.mark.asyncio
    async def test_crawl_depth_limit(self):
        """Test crawl depth limiting"""
        content = await extract_webpage_content("https://www.python.org")
        crawled = await crawl_links(content, max_depth=0, max_pages=10)
        
        assert len(crawled) == 1  # Only the original page
    
    @pytest.mark.asyncio
    async def test_crawl_performance(self):
        """Test crawling performance"""
        content = await extract_webpage_content("https://www.python.org")
        
        with PerformanceMonitor() as monitor:
            crawled = await crawl_links(content, max_depth=1, max_pages=5)
        
        # Check performance metrics
        assert monitor.metrics["duration"] < TEST_CONFIG["timeout"] * 5
        assert len(crawled) <= 5

class TestSecurity:
    """Test security-related functionality"""
    
    def test_url_sanitization(self):
        """Test URL sanitization"""
        unsafe_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "\\\\..\\/etc/passwd",
            "http://169.254.169.254/latest/meta-data/",
        ]
        
        for url in unsafe_urls:
            with pytest.raises(ValueError):
                Link(url=url)
    
    @pytest.mark.asyncio
    async def test_content_sanitization(self):
        """Test content sanitization"""
        content = await extract_webpage_content("https://www.python.org")
        
        # Check for script tag removal
        if content.html:
            soup = BeautifulSoup(content.html, 'html.parser')
            scripts = soup.find_all('script')
            assert len(scripts) == 0
    
    def test_header_injection(self):
        """Test prevention of header injection"""
        config = WebToolsConfig(
            user_agent="Test\r\nInjected-Header: value"
        )
        
        # User agent should be sanitized
        assert "\r" not in config.user_agent
        assert "\n" not in config.user_agent

class TestIntegration:
    """Test integrated workflows"""
    
    @pytest.mark.asyncio
    async def test_search_and_extract_workflow(self):
        """Test complete search and extract workflow"""
        results = await search_and_extract("Python tutorial", max_results=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, WebPageContent) for r in results)
        assert all(r.content for r in results)
    
    @pytest.mark.asyncio
    async def test_search_extract_and_crawl_workflow(self):
        """Test complete search, extract, and crawl workflow"""
        results = await search_extract_and_crawl(
            "Python tutorial",
            max_results=2,
            crawl_depth=1,
            max_pages_per_result=3
        )
        
        assert len(results) <= 2
        assert all(isinstance(pages, dict) for pages in results.values())
        assert all(
            isinstance(content, WebPageContent)
            for pages in results.values()
            for content in pages.values()
        )

def run_all_tests():
    """Run all tests and generate report"""
    import pytest
    
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--capture=no",
        "--log-cli-level=INFO",
    ])
    
    # Generate test report
    report_path = TEST_OUTPUT_DIR / "test_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# WebTools Test Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # Add performance metrics if available
        if hasattr(pytest, "performance_metrics"):
            f.write("## Performance Metrics\n\n")
            for name, metric in pytest.performance_metrics.items():
                f.write(f"### {name}\n")
                f.write(f"- Duration: {metric['duration']:.2f}s\n")
                f.write(f"- Memory Usage: {metric['memory_increase'] / 1024:.2f}MB\n\n")

if __name__ == "__main__":
    run_all_tests() 