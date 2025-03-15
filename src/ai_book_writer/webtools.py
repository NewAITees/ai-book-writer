#!/usr/bin/env python3
"""
Web scraping and search tools using Crawl4AI.
This module provides a clean interface for retrieving information from the web.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import json

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote

# ロギング設定
logger = logging.getLogger("webtools")

# JSONシリアライズ用のヘルパー関数
def debug_crawl_result(result, url="unknown"):
    """クロール結果のデバッグ情報を出力"""
    logger.info(f"===== クロール結果デバッグ [{url}] =====")
    logger.info(f"オブジェクトタイプ: {type(result)}")
    logger.info(f"利用可能な属性: {dir(result)}")
    
    # 主要属性の存在チェックと型確認
    for attr in ['text', 'markdown', 'html', 'metadata', 'links', 'images', 'children']:
        if hasattr(result, attr):
            attr_value = getattr(result, attr)
            logger.info(f"属性 '{attr}' が存在: {type(attr_value)}")
            
            # リストや辞書の場合は要素数も出力
            if isinstance(attr_value, (list, dict)):
                logger.info(f"  - 要素数: {len(attr_value)}")
            
            # テキスト属性はサンプルを出力
            if attr in ['text', 'markdown', 'html'] and isinstance(attr_value, str):
                logger.info(f"  - サンプル: {attr_value[:100]}...")
        else:
            logger.info(f"属性 '{attr}' は存在しません")
    
    logger.info("=====================================")

def get_safe_attribute(obj, primary_attr, fallback_attrs=None, default=None):
    """安全に属性にアクセスする。主属性がなければフォールバック属性を試す"""
    if hasattr(obj, primary_attr):
        return getattr(obj, primary_attr)
    
    if fallback_attrs:
        for attr in fallback_attrs:
            if "." in attr:  # ネストした属性アクセス (例: "metadata.title")
                parts = attr.split(".")
                current = obj
                found = True
                for part in parts:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        found = False
                        break
                if found:
                    return current
            elif hasattr(obj, attr):
                return getattr(obj, attr)
    
    return default

class CrawlResultEncoder(json.JSONEncoder):
    """Crawl4AI結果オブジェクト用のJSONエンコーダ"""
    def default(self, obj):
        try:
            # 基本的なシリアライズを試みる
            return super().default(obj)
        except TypeError:
            try:
                # 辞書変換を試みる
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                # 文字列変換を試みる
                return str(obj)
            except:
                # どうしてもダメな場合は代替表現
                return f"<非シリアライズ可能オブジェクト: {type(obj).__name__}>"

def process_crawl_result(result, url):
    """クロール結果を処理してJSONシリアライズ可能な辞書を生成"""
    # デバッグ情報を出力
    debug_crawl_result(result, url)
    
    # 基本情報を設定
    result_info = {
        "url": url,
        "title": get_safe_attribute(result, 'title', 
                 ['metadata.title'], 'No title'),
        "crawled_success": True,
    }
    
    # コンテンツ関連の属性を安全に取得
    text_content = get_safe_attribute(result, 'text', 
                  ['content', 'body', 'raw_text'], '')
    if text_content:
        result_info["content_text"] = text_content
    
    markdown_content = get_safe_attribute(result, 'markdown', 
                      ['markdown_content', 'md'], '')
    if markdown_content:
        result_info["content_markdown"] = markdown_content
    
    html_content = get_safe_attribute(result, 'html', 
                  ['html_content', 'raw_html'], '')
    if html_content:
        result_info["content_html"] = html_content
    
    # メタデータの処理
    metadata = get_safe_attribute(result, 'metadata', None, {})
    if metadata:
        try:
            # メタデータが直接シリアライズできない場合は辞書に変換
            if not isinstance(metadata, dict):
                if hasattr(metadata, '__dict__'):
                    metadata = metadata.__dict__
                else:
                    metadata = {"value": str(metadata)}
            result_info["metadata"] = metadata
        except:
            result_info["metadata"] = {"error": "メタデータをシリアライズできません"}
    
    # リンク情報の処理
    links = get_safe_attribute(result, 'links', None, [])
    if links:
        try:
            if isinstance(links, list):
                result_info["links"] = links
            elif isinstance(links, dict):
                result_info["links"] = {
                    "external": links.get('external', []),
                    "internal": links.get('internal', [])
                }
            else:
                result_info["links"] = str(links)
            result_info["links_count"] = len(links)
        except:
            result_info["links"] = "リンク情報を取得できません"
            result_info["links_count"] = 0
    
    # 子ページの処理
    children = get_safe_attribute(result, 'children', None, [])
    if children:
        try:
            child_pages = []
            for child in children:
                child_url = get_safe_attribute(child, 'url', None, "Unknown URL")
                child_info = {
                    "url": child_url,
                    "title": get_safe_attribute(child, 'title', 
                             ['metadata.title'], 'No title'),
                }
                
                # 子ページのコンテンツを取得
                child_text = get_safe_attribute(child, 'text', None, '')
                if child_text:
                    child_info["content_text"] = child_text
                
                child_markdown = get_safe_attribute(child, 'markdown', None, '')
                if child_markdown:
                    child_info["content_markdown"] = child_markdown
                
                child_pages.append(child_info)
            
            result_info["child_pages"] = child_pages
        except Exception as e:
            result_info["child_pages_error"] = str(e)
    
    return result_info

def save_results_to_json(results_data, save_file):
    """結果データをJSONファイルに保存"""
    try:
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2, cls=CrawlResultEncoder)
        logger.info(f"結果を {save_file} に保存しました")
        return True
    except Exception as e:
        logger.error(f"JSONファイル保存中にエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # エラーが発生した場合、より単純な形式で再試行
        try:
            # 問題のあるデータを特定するためにキーごとに保存を試みる
            problem_keys = []
            for key, value in results_data.items():
                try:
                    test_json = json.dumps({key: value}, ensure_ascii=False, cls=CrawlResultEncoder)
                except Exception:
                    problem_keys.append(key)
            
            logger.warning(f"問題のあるキー: {problem_keys}")
            
            # 問題のあるキーを除外して保存
            safe_data = {k: v for k, v in results_data.items() if k not in problem_keys}
            with open(f"{save_file}.safe.json", 'w', encoding='utf-8') as f:
                json.dump(safe_data, f, ensure_ascii=False, indent=2)
            logger.info(f"問題のあるキーを除外して {save_file}.safe.json に保存しました")
        except Exception as e2:
            logger.error(f"安全保存中にも失敗: {str(e2)}")
        
        return False

# リンク情報を表すモデルを追加
class LinkInfo(BaseModel):
    """リンク情報の構造"""
    href: str  # リンクURL (必須)
    domain: Optional[str] = None  # ドメイン情報 (オプション)
    text: Optional[str] = None  # リンクテキスト (オプション)
    url: Optional[str] = None  # 代替URL表現 (オプション)

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
    # 複数の型を許容するようにUnionを使用
    links: List[Union[str, LinkInfo, Dict[str, Any]]] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)

def normalize_links(links_data: Any) -> List[Union[str, LinkInfo, Dict[str, Any]]]:
    """
    リンクデータを正規化する関数
    
    Args:
        links_data: crawl4aiから返されるリンク情報
        
    Returns:
        List[Union[str, LinkInfo, Dict[str, Any]]]: 正規化されたリンクリスト
    """
    if not links_data:
        return []
    
    normalized_links = []
    
    # リストの場合
    if isinstance(links_data, list):
        for link in links_data:
            # 文字列の場合はそのまま追加
            if isinstance(link, str):
                normalized_links.append(link)
            # 辞書の場合はLinkInfoに変換を試みる
            elif isinstance(link, dict):
                # href属性がなくてもurl属性があれば対応
                if 'href' in link:
                    link_dict = link.copy()
                elif 'url' in link:
                    link_dict = link.copy()
                    link_dict['href'] = link['url']
                else:
                    # 必須属性がない辞書は無視
                    continue
                    
                # LinkInfoモデルへの変換を試みる
                try:
                    normalized_links.append(LinkInfo(**link_dict))
                except Exception as e:
                    logger.debug(f"LinkInfo変換エラー: {e}")
                    # 変換に失敗した場合は辞書のまま追加
                    normalized_links.append(link)
    
    # 辞書の場合 (internal/externalに分かれている場合)
    elif isinstance(links_data, dict):
        for link_type, links in links_data.items():
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, str):
                        normalized_links.append(link)
                    elif isinstance(link, dict):
                        # href属性がなくてもurl属性があれば対応
                        if 'href' in link:
                            link_dict = link.copy()
                        elif 'url' in link:
                            link_dict = link.copy()
                            link_dict['href'] = link['url']
                        else:
                            # 必須属性がない辞書は無視
                            continue
                            
                        try:
                            normalized_links.append(LinkInfo(**link_dict))
                        except Exception:
                            normalized_links.append(link)
    
    return normalized_links

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
        from duckduckgo_search import DDGS
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
    from urllib.parse import quote_plus
    sample_data = {
        "AbstractURL": f"https://example.com/about/{quote_plus(query)}",
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
                    
                    # デバッグ情報を出力
                    logger.info(f"結果オブジェクトの型: {type(result)}")
                    logger.info(f"利用可能な属性: {dir(result)}")
                    
                    # 新しい処理関数を使用
                    result_info = process_crawl_result(result, url)
                    result_info["index"] = i
                    result_info["title"] = url_data["title"]  # 元のタイトルを優先
                    
                    results_data["results"].append(result_info)
                    
                except Exception as e:
                    logger.error(f"URL {url} のクロール中にエラー: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    results_data["results"].append({
                        "index": i,
                        "url": url,
                        "title": url_data["title"],
                        "crawled_success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
        
        # 結果を表示
        logger.info(f"クロール完了: {len(results_data['results'])}件の結果")
        
        # 修正した保存関数を使用
        if save_file:
            save_results_to_json(results_data, save_file)
        
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

async def run_search_tests(queries=None, output_dir=None):
    """
    複数の検索クエリをテスト実行
    
    Args:
        queries: テストする検索クエリのリスト
        output_dir: 結果を保存するディレクトリ
    """
    if queries is None:
        queries = [
            "Python machine learning",
            "AI framework",
            "latest technology trends"
        ]
    
    if output_dir:
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    for query in queries:
        logger.info(f"\n===== クエリ '{query}' のテスト =====")
        
        save_file = None
        if output_dir:
            save_file = output_dir / f"search_results_{query.replace(' ', '_')}.json"
            
        results = await test_search_with_deep_crawling(
            query, 
            num_results=5,
            save_file=save_file,
            use_fallback=True  # APIが失敗した場合はフォールバックを使用
        )
        
        # 検索結果があれば最初のURLをテスト
        if results and "links" in results and results["links"]:
            first_link = results["links"][0]
            if "url" in first_link:
                logger.info(f"\n===== 最初のURLのフェッチテスト =====")
                await test_direct_url_fetch(first_link["url"])

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
        # URLの検証
        if not url or not isinstance(url, str) or not (url.startswith('http://') or url.startswith('https://')):
            logger.error(f"Invalid URL format: {url}")
            return WebContent(
                url=str(url) if url else "",
                title="Invalid URL",
                content="Error: Invalid URL format",
                markdown="# Error\n\nFailed to fetch content: Invalid URL format",
                links=[],
                images=[]
            )
            
        async with AsyncWebCrawler() as crawler:
            browser_config = kwargs.get('browser_config', BrowserConfig(
                headless=True,
                ignore_https_errors=True
            ))
            run_config = kwargs.get('run_config', CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS
            ))
            
            result = await crawler.arun(
                url=url, 
                browser_config=browser_config,
                run_config=run_config
            )
            
            # デバッグ情報の出力（詳細な問題分析用）
            debug_crawl_result(result, url)
            
            # 安全な属性アクセスを使用
            title = get_safe_attribute(result, 'title', ['metadata.title'], '')
            text_content = get_safe_attribute(result, 'text', ['content', 'body'], '')
            markdown_content = get_safe_attribute(result, 'markdown', ['markdown_content', 'md'], '')
            html_content = get_safe_attribute(result, 'html', ['html_content', 'raw_html'], '')
            
            # リンクとイメージの安全な取得
            links_data = get_safe_attribute(result, 'links', None, [])
            images_data = get_safe_attribute(result, 'images', None, [])
            
            # リンク情報の処理
            if links_data:
                # リンクデータの構造をログに出力（デバッグ用）
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Links type: {type(links_data)}")
                    if isinstance(links_data, list) and links_data:
                        logger.debug(f"First link example: {links_data[0]}")
                    elif isinstance(links_data, dict):
                        for key, val in links_data.items():
                            if val and isinstance(val, list) and val:
                                logger.debug(f"First {key} link example: {val[0]}")
            
            # リンク情報を正規化
            links = normalize_links(links_data)
            
            # 画像情報の処理
            images = []
            if isinstance(images_data, list):
                images = images_data
            
            # WebContentオブジェクト作成（Pydanticモデルなのでシリアライズ可能）
            return WebContent(
                url=url,
                title=title,
                content=text_content,
                markdown=markdown_content,
                html=html_content,
                links=links,
                images=images
            )
    except Exception as e:
        logger.error(f"Error fetching webpage {url}: {str(e)}")
        import traceback
        logger.error(f"詳細なエラー: {traceback.format_exc()}")
        
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

def clean_url(url):
    """URLをクリーンアップする"""
    if not url or not isinstance(url, str):
        return None
    
    # 相対URLの場合はスキップ
    if not url.startswith('http://') and not url.startswith('https://'):
        return None
    
    # Googleの内部リンクはスキップ
    if 'google.com' in url or 'gstatic.com' in url or 'googleapis.com' in url:
        return None
    
    return url

def extract_links_from_google_news(html_content, base_url):
    """Google ニュースのHTMLからリンクを抽出"""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    
    # カードやニュース記事内のリンクを探す
    for a_tag in soup.find_all('a', href=True):
        url = a_tag.get('href', '')
        text = a_tag.get_text(strip=True) or ''
        
        # 相対URLを絶対URLに変換
        if url.startswith('/'):
            url = urljoin(base_url, url)
        
        # GoogleリダイレクトURLがある場合は元のURLを抽出
        if '/url?q=' in url:
            # URLから元のリンクを抽出
            try:
                import re
                match = re.search(r'/url\?q=([^&]+)', url)
                if match:
                    url = match.group(1)
            except:
                pass
        
        url = clean_url(url)
        if url and len(text) > 0:  # テキストがあるリンクのみ
            links.append({
                'url': url,
                'text': text[:100]  # テキストは長すぎる場合があるので制限
            })
    
    return links

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
        logger.info(f"検索クエリ実行: {query}")
        
        # DuckDuckGo APIで検索を実行
        search_results = await duckduckgo_search(query, max_results=num_results*2)  # 余裕を持って多めに取得
        
        # 検索に失敗した場合はフォールバック
        if not search_results:
            logger.warning("DuckDuckGo検索に失敗したため、フォールバックを使用します")
            search_results = await google_search_simulation(query, max_results=num_results*2)
        
        if not search_results:
            logger.error(f"検索 '{query}' の結果が取得できませんでした")
            return []
        
        # 検索結果からURLを抽出
        urls_to_crawl = []
        
        # AbstractURL（主要な結果URL）があれば追加
        if search_results.get("AbstractURL"):
            urls_to_crawl.append(search_results["AbstractURL"])
        
        # Resultsセクションから結果を追加
        if search_results.get("Results"):
            for result in search_results["Results"]:
                if "FirstURL" in result:
                    urls_to_crawl.append(result["FirstURL"])
        
        # RelatedTopicsから関連URLを追加
        if search_results.get("RelatedTopics"):
            for topic in search_results["RelatedTopics"]:
                if "FirstURL" in topic:
                    urls_to_crawl.append(topic["FirstURL"])
        
        # 重複を除去
        unique_urls = []
        for url in urls_to_crawl:
            if url not in unique_urls:
                unique_urls.append(url)
        
        if not unique_urls:
            logger.warning(f"検索 '{query}' の結果からURLを抽出できませんでした")
            return []
        
        logger.info(f"抽出されたURL数: {len(unique_urls)}")
        
        # 各リンクの内容を非同期で取得
        valid_contents = []
        tasks = []
        
        for url in unique_urls[:num_results]:  # 指定された数だけ処理
            if url and isinstance(url, str) and (url.startswith('http://') or url.startswith('https://')):
                tasks.append(fetch_webpage_content_async(url))
        
        if tasks:
            # エラーハンドリングを改善した非同期処理
            contents = []
            for task in asyncio.as_completed(tasks):
                try:
                    content = await task
                    if isinstance(content, WebContent):
                        valid_contents.append(content)
                        logger.info(f"コンテンツを取得しました: {content.url} (タイトル: {content.title[:30]}...)")
                except Exception as e:
                    logger.error(f"コンテンツ取得中にエラー: {str(e)}")
                    import traceback
                    logger.debug(f"コンテンツ取得エラーの詳細: {traceback.format_exc()}")
        
        logger.info(f"取得したコンテンツ数: {len(valid_contents)}")
        
        # 収集したコンテンツを確認
        for i, content in enumerate(valid_contents):
            logger.info(f"コンテンツ {i+1}:")
            logger.info(f"  URL: {content.url}")
            logger.info(f"  タイトル: {content.title[:50]}...")
            logger.info(f"  テキスト長: {len(content.content)} 文字")
            logger.info(f"  マークダウン長: {len(content.markdown)} 文字")
        
        return valid_contents
    
    except Exception as e:
        logger.error(f"検索と抽出中にエラーが発生しました: {str(e)}")
        import traceback
        logger.error(f"詳細なエラー: {traceback.format_exc()}")
        return []

async def try_crawl4ai_approach(search_url: str, num_results: int = 5) -> List[str]:
    """Crawl4AIを使用した検索リンク抽出を試みる"""
    try:
        # ブラウザ設定
        browser_config = BrowserConfig(
            headless=True,
            ignore_https_errors=True
        )
        
        # 深いクローリング戦略を使用
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=1,
                include_external=True,
                max_pages=num_results * 2  # 余裕を持って多めに指定
            ),
            verbose=True
        )
        
        # 検索結果ページからリンクを抽出
        logger.info(f"Crawl4AIで検索ページをクロール: {search_url}")
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=search_url,
                browser_config=browser_config,
                run_config=run_config
            )
            
            # デバッグ情報出力
            debug_crawl_result(result, search_url)
            
            # リンクを抽出
            links = []
            try:
                links_data = get_safe_attribute(result, 'links', None, [])
                
                if isinstance(links_data, list):
                    links = links_data
                elif isinstance(links_data, dict):
                    if 'external' in links_data:
                        external = links_data.get('external', [])
                        if isinstance(external, list):
                            links.extend(external)
                    if 'internal' in links_data:
                        # 内部リンクからGoogle以外のドメインのみ抽出
                        internal = links_data.get('internal', [])
                        if isinstance(internal, list):
                            for link in internal:
                                if link and isinstance(link, str) and not any(domain in link for domain in ['google.', 'gstatic.', 'googleapis.']):
                                    links.append(link)
            except Exception as e:
                logger.error(f"リンク抽出中にエラー: {str(e)}")
                
            # 検索結果らしきリンクのみ抽出（単純フィルタリング）
            filtered_links = []
            for link in links:
                try:
                    if link and isinstance(link, str) and (link.startswith('http://') or link.startswith('https://')):
                        # Googleの内部リンクはスキップ
                        if not any(domain in link for domain in ['google.', 'gstatic.', 'googleapis.']):
                            filtered_links.append(link)
                except Exception as e:
                    logger.warning(f"リンクフィルタリング中にエラー: {str(e)}")
            
            # 重複を削除
            unique_links = []
            link_set = set()
            for link in filtered_links:
                if link not in link_set:
                    link_set.add(link)
                    unique_links.append(link)
            
            logger.info(f"Crawl4AIアプローチで {len(unique_links)} 個のリンクを抽出しました")
            
            # リンクの一部をログ出力
            for i, link in enumerate(unique_links[:5]):
                logger.debug(f"抽出リンク {i+1}: {link}")
            
            # 子ページを取得する場合
            child_urls = []
            try:
                children = get_safe_attribute(result, 'children', None, [])
                if children and isinstance(children, list):
                    for child in children:
                        child_url = get_safe_attribute(child, 'url', None, None)
                        if child_url:
                            child_urls.append(child_url)
                    
                    logger.info(f"ディープクローリングで {len(child_urls)} 個の子ページURLを抽出しました")
                    
                    # 子ページURLもフィルタリング
                    filtered_child_urls = []
                    for url in child_urls:
                        if url and isinstance(url, str) and (url.startswith('http://') or url.startswith('https://')):
                            if not any(domain in url for domain in ['google.', 'gstatic.', 'googleapis.']):
                                if url not in link_set:  # 上記のリンク集合にないもののみ
                                    filtered_child_urls.append(url)
                    
                    # 結果リストに追加
                    unique_links.extend(filtered_child_urls)
            except Exception as e:
                logger.error(f"子ページ処理中にエラー: {str(e)}")
            
            # 最終的なリンクを返す（指定された数まで）
            return unique_links[:num_results]
                
        return []
    except Exception as e:
        logger.error(f"Crawl4AIアプローチでエラー: {str(e)}")
        import traceback
        logger.error(f"詳細なエラー: {traceback.format_exc()}")
        return []

def try_browserless_approach(search_url: str, num_results: int = 5) -> List[dict]:
    """ブラウザレスなアプローチによる検索リンク抽出を試みる"""
    try:
        # ユーザーエージェントを設定
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        # 検索ページを取得
        logger.info(f"ブラウザレスアプローチで検索ページを取得: {search_url}")
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # エンコーディングを適切に設定
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding
        
        # 検索結果のURLがGoogle Newsの場合
        if 'news.google.com' in search_url:
            # Googleニュース専用のリンク抽出
            links = extract_links_from_google_news(response.text, search_url)
            logger.info(f"Googleニュースから {len(links)} 個のリンクを抽出しました")
        else:
            # 通常のリンク抽出
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            # リンクを抽出（単純なaタグの取得）
            for a_tag in soup.find_all('a', href=True):
                try:
                    url = a_tag.get('href', '')
                    text = a_tag.get_text(strip=True) or ''
                    
                    # 相対URLを絶対URLに変換
                    if url.startswith('/'):
                        url = urljoin(search_url, url)
                    
                    # GoogleリダイレクトURLがある場合は元のURLを抽出
                    if '/url?q=' in url:
                        try:
                            import re
                            match = re.search(r'/url\?q=([^&]+)', url)
                            if match:
                                url = match.group(1)
                        except Exception as regex_e:
                            logger.warning(f"リダイレクトURL解析中にエラー: {str(regex_e)}")
                            continue
                    
                    # URLを安全にクリーンアップ
                    url = clean_url(url)
                    if url and len(text) > 0:  # テキストがあるリンクのみ
                        links.append({
                            'url': url,
                            'text': text[:100] if text else ''  # テキストは長すぎる場合があるので制限
                        })
                except Exception as link_e:
                    logger.warning(f"リンク抽出中に個別エラー: {str(link_e)}")
                    continue
        
        # リンクの基本情報をログ出力
        logger.info(f"抽出された生のリンク数: {len(links)}")
        
        # 重複を除去して上位の結果だけを返す
        unique_urls = set()
        filtered_links = []
        
        for link in links:
            try:
                if isinstance(link, dict) and 'url' in link:
                    url = link['url']
                    text = link.get('text', '')
                elif isinstance(link, str):
                    url = link
                    text = ''
                    link = {'url': url, 'text': text}
                else:
                    continue
                    
                # URLの検証と重複チェック
                if url and isinstance(url, str) and url not in unique_urls:
                    # 安全なスキーマチェック
                    if url.startswith('http://') or url.startswith('https://'):
                        unique_urls.add(url)
                        # リンクテキストがない場合はURLの一部を使用
                        if not text:
                            parsed_url = urlparse(url)
                            text = parsed_url.netloc
                            if len(text) > 30:
                                text = text[:30] + '...'
                        
                        # 完全なリンク情報を追加
                        filtered_links.append({
                            'url': url, 
                            'text': text,
                            'domain': urlparse(url).netloc
                        })
                        
                        # 十分な数を集めたら終了
                        if len(filtered_links) >= num_results:
                            break
            except Exception as filter_e:
                logger.warning(f"リンクフィルタリング中にエラー: {str(filter_e)}")
        
        logger.info(f"フィルタリング後のリンク数: {len(filtered_links)}")
        
        # 取得したリンクの一部をログ出力
        for i, link in enumerate(filtered_links[:3]):
            logger.info(f"抽出されたリンク {i+1}: {link['url']} - {link['text'][:30]}")
        
        return filtered_links
    except Exception as e:
        logger.error(f"ブラウザレスアプローチでエラー: {str(e)}")
        import traceback
        logger.error(f"詳細なエラー: {traceback.format_exc()}")
        return []

async def test_crawl4ai_result_structure(url: str = "https://www.python.org/"):
    """
    crawl4aiライブラリの結果構造を詳細に調査するためのテスト関数
    
    Args:
        url: テスト対象のURL
    """
    logger.info(f"Crawl4AIの結果構造テスト: {url}")
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            # 基本情報の出力
            logger.info(f"結果オブジェクトの型: {type(result)}")
            logger.info(f"利用可能な属性: {dir(result)}")
            
            # 重要な属性を詳細に調査
            for attr_name in ['text', 'markdown', 'html', 'metadata', 'links', 'images']:
                if hasattr(result, attr_name):
                    attr_value = getattr(result, attr_name)
                    logger.info(f"属性 '{attr_name}' の型: {type(attr_value)}")
                    
                    # コレクションの場合は内容も確認
                    if isinstance(attr_value, (list, tuple)) and attr_value:
                        logger.info(f"  - 要素数: {len(attr_value)}")
                        logger.info(f"  - 最初の要素の型: {type(attr_value[0])}")
                        logger.info(f"  - 最初の要素の内容: {attr_value[0]}")
                    elif isinstance(attr_value, dict) and attr_value:
                        logger.info(f"  - キー数: {len(attr_value)}")
                        logger.info(f"  - キー一覧: {list(attr_value.keys())}")
                        
                        # 特に links 属性は詳細に調査
                        if attr_name == 'links':
                            for key, val in attr_value.items():
                                if isinstance(val, list) and val:
                                    logger.info(f"  - {key} の最初の要素の型: {type(val[0])}")
                                    logger.info(f"  - {key} の最初の要素の内容: {val[0]}")
                else:
                    logger.info(f"属性 '{attr_name}' は存在しません")
            
            # リンク情報を正規化してテスト
            if hasattr(result, 'links'):
                normalized = normalize_links(result.links)
                logger.info(f"正規化されたリンク数: {len(normalized)}")
                if normalized:
                    logger.info(f"正規化された最初のリンク: {normalized[0]}")
                    
                # WebContentクラスで変換テスト
                try:
                    content = WebContent(
                        url=url,
                        title=get_safe_attribute(result, 'title', ['metadata.title'], 'Test Page'),
                        content=get_safe_attribute(result, 'text', [], 'Test Content'),
                        markdown=get_safe_attribute(result, 'markdown', [], 'Test Markdown'),
                        links=normalized
                    )
                    logger.info("WebContentへの変換成功")
                except Exception as e:
                    logger.error(f"WebContentへの変換失敗: {e}")
            
            return result
            
    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# メインの部分（テスト用）
if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # コマンドライン引数を処理
    import sys
    
    # 特別なテスト引数のチェック
    if len(sys.argv) > 1 and sys.argv[1] == "--test-crawl4ai-structure":
        # 構造テスト用のURL指定
        test_url = "https://www.python.org/"
        if len(sys.argv) > 2:
            test_url = sys.argv[2]
        
        print(f"Crawl4AI結果構造テスト実行: {test_url}")
        asyncio.run(test_crawl4ai_result_structure(test_url))
        sys.exit(0)
        
    # 通常のコマンドライン引数処理
    queries = []
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        queries.append(query)
        logger.info(f"コマンドライン引数からクエリを追加: '{query}'")
    
    # テスト実行
    if queries:
        asyncio.run(run_search_tests(queries=queries, output_dir="search_results"))
    else:
        # デフォルトのテスト
        url = "https://www.python.org/"
        print(f"Fetching content from: {url}")
        
        content = fetch_webpage_content(url)
        
        print(f"Title: {content.title}")
        print(f"Content length: {len(content.content)} characters")
        print(f"Markdown length: {len(content.markdown)} characters")
        print(f"Sample markdown: {content.markdown[:300]}...")
        print(f"Found {len(content.links)} links and {len(content.images)} images")
        
        # リンク情報の詳細出力
        if content.links:
            print("\nリンク情報サンプル:")
            for i, link in enumerate(content.links[:3]):
                print(f"リンク {i+1}: {link}")
        
        # 検索と内容抽出のテスト
        async def test_search():
            query = "Python programming language"
            print(f"\nSearching for: {query}")
            
            results = await search_and_extract_content(query, 3)
            
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result.title}")
                print(f"   Content length: {len(result.content)} characters")
                print(f"   Links count: {len(result.links)}")
                print(f"   URL: {result.url}\n")
        
        asyncio.run(test_search()) 