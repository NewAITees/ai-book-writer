#!/usr/bin/env python3
"""
ブラウザレスなウェブ検索とリンク抽出のテスト
"""

import asyncio
import logging
import json
import pprint
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_browserless")

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

def fetch_webpage_content(url):
    """
    ウェブページのコンテンツを取得する
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # エラーチェック
        
        # エンコーディングを適切に設定
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding
        
        return {
            'url': url,
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'text': response.text
        }
    except Exception as e:
        logger.error(f"URLフェッチエラー ({url}): {str(e)}")
        return None

def run_google_news_search(query, num_results=5, save_file=None):
    """
    Google Newsを使って検索結果とリンクを取得する
    """
    try:
        # 検索URLを設定
        encoded_query = quote(query)
        search_url = f"https://news.google.com/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
        logger.info(f"検索URL: {search_url}")
        
        # ページを取得
        logger.info("ページ取得開始...")
        page_content = fetch_webpage_content(search_url)
        
        if not page_content or not page_content.get('text'):
            logger.error("ページ内容を取得できませんでした")
            return None
        
        # リンクを抽出
        links = extract_links_from_google_news(page_content['text'], search_url)
        logger.info(f"全リンク数: {len(links)}")
        
        # 重複を除去して上位の結果だけを返す
        unique_urls = set()
        filtered_links = []
        
        for link in links:
            url = link['url']
            if url and url not in unique_urls:
                unique_urls.add(url)
                filtered_links.append(link)
                if len(filtered_links) >= num_results:
                    break
        
        # 結果をまとめる
        results = {
            'query': query,
            'search_url': search_url,
            'total_links_found': len(links),
            'filtered_links': filtered_links
        }
        
        # 最初の数件のリンクを表示
        logger.info(f"フィルタリング後のリンク数: {len(filtered_links)}")
        for i, link in enumerate(filtered_links[:5]):
            logger.info(f"リンク {i+1}: {link['url']} - {link['text']}")
        
        # ファイルに保存
        if save_file:
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"結果を {save_file} に保存しました")
        
        return results
    
    except Exception as e:
        logger.error(f"検索中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    メイン関数
    """
    # 出力ディレクトリの作成
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # 検索テスト
    queries = [
        "Python 機械学習",
        "AIフレームワーク比較",
        "最新テクノロジー"
    ]
    
    for query in queries:
        logger.info(f"\n===== クエリ '{query}' のテスト =====")
        results = run_google_news_search(
            query, 
            num_results=5,
            save_file=output_dir / f"browserless_{query.replace(' ', '_')}.json"
        )
        
        # リンクの最初のURLの内容を取得
        if results and results.get('filtered_links') and len(results['filtered_links']) > 0:
            first_url = results['filtered_links'][0]['url']
            
            logger.info(f"\n===== 最初のURL '{first_url}' のコンテンツ取得 =====")
            content = fetch_webpage_content(first_url)
            
            if content:
                logger.info(f"ステータスコード: {content['status_code']}")
                logger.info(f"コンテンツタイプ: {content['content_type']}")
                
                # HTMLのサイズを表示
                html_size = len(content['text'])
                logger.info(f"HTML サイズ: {html_size} バイト")
                
                # 最初の500文字だけ表示
                preview = content['text'][:500].replace('\n', ' ')
                logger.info(f"プレビュー: {preview}...")

if __name__ == "__main__":
    main() 