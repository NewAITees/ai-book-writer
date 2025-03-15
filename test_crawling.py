#!/usr/bin/env python3
"""
Crawl4AIを使ったDuckDuckGo検索とリンク抽出のテスト
"""

import asyncio
import logging
import json
import pprint
import os
import urllib.parse
from pathlib import Path
import aiohttp
from duckduckgo_search import DDGS

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_crawling")

async def duckduckgo_search(query: str, max_results: int = 10):
    """
    DuckDuckGoのAPIを使用して検索を実行
    
    Args:
        query: 検索クエリ
        max_results: 取得する結果の最大数
        
    Returns:
        dict: 検索結果
    """
    try:
        logger.info(f"DuckDuckGo検索実行: クエリ '{query}'")
        
        # DDGSクラスを使用して検索を実行
        # DDGSはsyncなクラスのため、非同期関数内でも同期的に実行
        loop = asyncio.get_running_loop()
        # run_in_executor内でwith文を使うためのラムダ関数を作成
        results = await loop.run_in_executor(None, lambda: list(DDGS().text(
            keywords=query,
            region='jp-jp',  # 日本の検索結果を優先
            safesearch='off',
            timelimit=None,
            max_results=max_results
        )))
            
        if not results:
            logger.warning("DuckDuckGo検索で結果が得られませんでした")
            return None
            
        logger.info(f"検索結果を取得しました: {len(results)}件")
        
        # DDGSの結果を元の関数の戻り値形式に変換
        formatted_results = {
            "AbstractURL": results[0]["href"] if results else "",
            "Heading": results[0]["title"] if results else "",
            "AbstractText": results[0]["body"] if results else "",
            "Results": [],
            "RelatedTopics": []
        }
        
        # 残りの結果をResultsに追加
        for i, result in enumerate(results):
            if i == 0:  # 最初の結果は既にAbstractURLとして追加済み
                continue
                
            formatted_results["Results"].append({
                "FirstURL": result["href"],
                "Text": result["title"]
            })
        
        # 結果の一部をログに出力（デバッグ用）
        logger.info(f"整形した検索結果の一部: {str(formatted_results)[:200]}...")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"DuckDuckGo検索中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def google_search_simulation(query: str, max_results: int = 5):
    """
    GoogleのAPIが使えない場合のシミュレーションデータ
    実際のAPIが使えない場合のフォールバックとして
    """
    logger.info(f"検索シミュレーション: '{query}'")
    
    # ダミーデータを生成
    sample_data = {
        "AbstractURL": f"https://example.com/about/{urllib.parse.quote_plus(query)}",
        "Heading": f"{query} に関する情報",
        "AbstractText": f"{query}に関する詳細情報。これはダミーのテキストです。",
        "Results": [
            {"FirstURL": "https://www.python.org", "Text": "Python公式サイト"}, 
            {"FirstURL": "https://www.tensorflow.org", "Text": "TensorFlow"}, 
            {"FirstURL": "https://scikit-learn.org", "Text": "Scikit-learn"}
        ],
        "RelatedTopics": [
            {"FirstURL": "https://en.wikipedia.org/wiki/Machine_learning", "Text": "機械学習 - Wikipedia"}, 
            {"FirstURL": "https://github.com/topics/ai", "Text": "AI - GitHub"}
        ]
    }
    
    # クエリによって少し内容を変える
    if "Python" in query:
        sample_data["Results"].append({"FirstURL": "https://numpy.org", "Text": "NumPy"})
        sample_data["Results"].append({"FirstURL": "https://pandas.pydata.org", "Text": "Pandas"})
    elif "AI" in query or "人工知能" in query:
        sample_data["Results"].append({"FirstURL": "https://openai.com", "Text": "OpenAI"})
        sample_data["Results"].append({"FirstURL": "https://huggingface.co", "Text": "Hugging Face"})
    
    return sample_data

async def test_search_with_deep_crawling(query: str, num_results: int = 5, save_file=None, use_fallback: bool = True):
    """
    DuckDuckGoとクローリングを組み合わせた検索テスト
    
    Args:
        query: 検索クエリ
        num_results: 取得する結果の数
        save_file: 結果を保存するJSONファイル名（オプション）
        use_fallback: フォールバックデータを使用するかどうか
        
    Returns:
        dict: 検索と各ページのコンテンツ情報を含む辞書
    """
    try:
        # DuckDuckGo APIで検索を実行
        used_fallback = False
        search_results = await duckduckgo_search(query, max_results=num_results)
        
        # APIが失敗したらフォールバックを使用
        if not search_results and use_fallback:
            logger.warning("DuckDuckGo APIからの取得に失敗したため、フォールバックデータを使用します")
            search_results = await google_search_simulation(query, max_results=num_results)
            used_fallback = True
        
        if not search_results:
            logger.error("検索結果が取得できませんでした")
            return None
        
        # 結果データの初期化
        results_data = {
            "query": query,
            "search_engine": "Fallback Simulation" if used_fallback else "DuckDuckGo",
            "results": [],
            "links": [],
            "raw_results": search_results
        }
        
        # 検索結果からURLを抽出
        urls_to_crawl = []
        
        # AbstractURL（主要な結果URL）があれば追加
        if search_results.get("AbstractURL"):
            urls_to_crawl.append({
                "url": search_results["AbstractURL"],
                "title": search_results.get("Heading", "Abstract Result")
            })
        
        # Resultsセクションから結果を追加
        if search_results.get("Results"):
            for result in search_results["Results"]:
                if "FirstURL" in result:
                    urls_to_crawl.append({
                        "url": result["FirstURL"],
                        "title": result.get("Text", "No title")
                    })
        
        # RelatedTopicsから関連URLを追加
        if search_results.get("RelatedTopics"):
            for topic in search_results["RelatedTopics"]:
                if "FirstURL" in topic:
                    urls_to_crawl.append({
                        "url": topic["FirstURL"],
                        "title": topic.get("Text", "No title")
                    })
        
        # 検索で見つかったURLを記録
        results_data["links"] = urls_to_crawl[:num_results]
        
        # URLが見つからなかった場合はエラーログを出力
        if not urls_to_crawl:
            logger.error("検索結果からURLを抽出できませんでした")
            logger.info(f"検索結果データ: {str(search_results)}")
            if not used_fallback and use_fallback:
                logger.warning("フォールバックデータを使用します")
                return await test_search_with_deep_crawling(query, num_results, save_file, use_fallback=True)
            return None
        
        logger.info(f"抽出されたURL数: {len(urls_to_crawl)}")
        
        # ブラウザ設定
        browser_config = BrowserConfig(
            headless=True,
            ignore_https_errors=True
        )
        
        # URLをクロールする
        logger.info(f"取得したURL数: {len(urls_to_crawl)}")
        
        async with AsyncWebCrawler() as crawler:
            for i, url_data in enumerate(urls_to_crawl[:num_results]):
                url = url_data["url"]
                logger.info(f"URL {i+1} をクロール中: {url}")
                
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    deep_crawl_strategy=BFSDeepCrawlStrategy(
                        max_depth=1,
                        include_external=True,
                        max_pages=5  # ページごとに限定された数のサブページをクロール
                    ),
                    verbose=True
                )
                
                try:
                    result = await crawler.arun(
                        url=url,
                        browser_config=browser_config,
                        run_config=run_config
                    )
                    
                    # 結果の処理
                    result_info = {
                        "index": i,
                        "url": url,
                        "title": url_data["title"],
                        "crawled_success": True,
                    }
                    
                    # ページコンテンツを保存
                    if hasattr(result, 'text'):
                        result_info["content_text"] = result.text
                    
                    if hasattr(result, 'markdown'):
                        result_info["content_markdown"] = result.markdown
                    
                    if hasattr(result, 'metadata') and result.metadata:
                        result_info["metadata"] = result.metadata
                    
                    # HTMLコンテンツが利用可能な場合は保存
                    if hasattr(result, 'html'):
                        result_info["content_html"] = result.html
                    
                    # リンクの情報を収集
                    result_links = []
                    if hasattr(result, 'links'):
                        if isinstance(result.links, list):
                            result_links = result.links
                            result_info["links"] = result_links
                        elif isinstance(result.links, dict):
                            result_info["links"] = {
                                "external": result.links.get('external', []),
                                "internal": result.links.get('internal', [])
                            }
                            if 'external' in result.links:
                                result_links.extend(result.links['external'])
                            if 'internal' in result.links:
                                result_links.extend(result.links['internal'])
                    
                    result_info["links_count"] = len(result_links)
                    
                    # 子ページのコンテンツも取得（ディープクロールした場合）
                    if hasattr(result, 'children') and result.children:
                        child_pages = []
                        for child_result in result.children:
                            child_page = {
                                "url": child_result.url if hasattr(child_result, 'url') else "Unknown URL",
                                "title": child_result.metadata.get('title', 'No title') if hasattr(child_result, 'metadata') else "No title",
                            }
                            
                            # 子ページのコンテンツを保存
                            if hasattr(child_result, 'text'):
                                child_page["content_text"] = child_result.text
                            
                            if hasattr(child_result, 'markdown'):
                                child_page["content_markdown"] = child_result.markdown
                            
                            child_pages.append(child_page)
                        
                        result_info["child_pages"] = child_pages
                    
                    results_data["results"].append(result_info)
                    
                except Exception as e:
                    logger.error(f"URL {url} のクロール中にエラー: {str(e)}")
                    results_data["results"].append({
                        "index": i,
                        "url": url,
                        "title": url_data["title"],
                        "crawled_success": False,
                        "error": str(e)
                    })
        
        # 結果を表示
        logger.info(f"クロール完了: {len(results_data['results'])}件の結果")
        
        # ファイルに保存
        if save_file:
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            logger.info(f"結果を {save_file} に保存しました")
        
        return results_data
        
    except Exception as e:
        logger.error(f"検索中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def test_direct_url_fetch(url):
    """
    単一URLのフェッチをテスト
    """
    logger.info(f"URLをフェッチ: {url}")
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            logger.info(f"タイトル: {result.metadata.get('title', 'No title')}")
            logger.info(f"テキスト長: {len(result.text) if hasattr(result, 'text') else 'N/A'}")
            logger.info(f"マークダウン長: {len(result.markdown) if hasattr(result, 'markdown') else 'N/A'}")
            
            return {
                "url": url,
                "title": result.metadata.get("title", "No title"),
                "markdown_sample": result.markdown[:300] if hasattr(result, 'markdown') else "N/A"
            }
    except Exception as e:
        logger.error(f"URLフェッチ中にエラーが発生しました: {str(e)}")
        return None

async def main():
    """
    メイン関数
    """
    # 出力ディレクトリの作成
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 検索テスト
    queries = [
        "Python machine learning",
        "AI framework",
        "latest technology trends"
    ]
    
    # コマンドライン引数として渡されたクエリがあれば追加
    import sys
    if len(sys.argv) > 1:
        additional_query = " ".join(sys.argv[1:])
        queries.append(additional_query)
        logger.info(f"コマンドライン引数からクエリを追加: '{additional_query}'")
    
    for query in queries:
        logger.info(f"\n===== クエリ '{query}' のテスト =====")
        results = await test_search_with_deep_crawling(
            query, 
            num_results=5,
            save_file=output_dir / f"search_results_{query.replace(' ', '_')}.json",
            use_fallback=True  # APIが失敗した場合はフォールバックを使用
        )
        
        # 検索結果があれば最初のURLをテスト
        if results and "links" in results and results["links"]:
            first_link = results["links"][0]
            if "url" in first_link:
                logger.info(f"\n===== 最初のURLのフェッチテスト =====")
                await test_direct_url_fetch(first_link["url"])

if __name__ == "__main__":
    asyncio.run(main()) 