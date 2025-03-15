#!/usr/bin/env python3
"""
Ollama API helpers for generating book content with local LLM models.
This module provides functions to interact with Ollama's API for text generation.
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union

import ollama
import requests

# ロギング設定
logger = logging.getLogger("ollama_helpers")

class OllamaConfig:
    """Ollama API設定"""
    def __init__(
        self, 
        model: str = "gemma3:4b", 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 1024,
        max_tokens: int = 4096,
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_predict = num_predict
        self.max_tokens = max_tokens
        self.base_url = base_url

async def generate_text_async(
    prompt: str, 
    ollama_config: OllamaConfig,
    system_prompt: Optional[str] = None,
    stream: bool = False
) -> str:
    """
    Ollamaモデルでテキストを非同期生成
    
    Args:
        prompt: 生成のプロンプト
        ollama_config: Ollama設定
        system_prompt: システムプロンプト (オプション)
        stream: ストリーミング生成を使用するか
        
    Returns:
        str: 生成されたテキスト
    """
    try:
        # HTTP APIクライアントを使用
        url = f"{ollama_config.base_url}/api/generate"
        
        # リクエストデータ
        data = {
            "model": ollama_config.model,
            "prompt": prompt,
            "temperature": ollama_config.temperature,
            "top_p": ollama_config.top_p,
            "top_k": ollama_config.top_k,
            "num_predict": ollama_config.num_predict,
            "stream": stream
        }
        
        # システムプロンプトが指定されている場合
        if system_prompt:
            data["system"] = system_prompt
        
        if stream:
            # ストリーミング生成（イベントごとに処理）
            full_response = ""
            async with requests.Session() as session:
                async with session.post(url, json=data, stream=True) as response:
                    if response.status_code != 200:
                        logger.error(f"Ollama API error: {response.status_code}")
                        return f"Error: Ollama API returned status code {response.status_code}"
                    
                    # イベントごとに処理
                    async for line in response.iter_lines():
                        if line:
                            try:
                                event = json.loads(line)
                                if "response" in event:
                                    full_response += event["response"]
                                if event.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode JSON from Ollama API: {line}")
                    
            return full_response
        else:
            # 非ストリーミング生成
            response = requests.post(url, json=data)
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                return f"Error: Ollama API returned status code {response.status_code}"
            
            result = response.json()
            return result["response"]
            
    except Exception as e:
        logger.error(f"Error generating text with Ollama: {str(e)}")
        return f"Error: {str(e)}"

def generate_text(
    prompt: str, 
    ollama_config: OllamaConfig,
    system_prompt: Optional[str] = None
) -> str:
    """
    OllamaモデルでテキストをSync生成（AsyncのWraper）
    
    Args:
        prompt: 生成のプロンプト
        ollama_config: Ollama設定
        system_prompt: システムプロンプト (オプション)
        
    Returns:
        str: 生成されたテキスト
    """
    # 非同期関数を同期的に実行
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # 既存のイベントループが実行中の場合、新しいループを作成
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        loop = new_loop
    
    try:
        return loop.run_until_complete(
            generate_text_async(prompt, ollama_config, system_prompt, stream=False)
        )
    finally:
        if loop != asyncio.get_event_loop():
            loop.close()

async def check_model_availability(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """
    指定したモデルがOllamaで利用可能かチェック
    
    Args:
        model_name: 確認するモデル名
        base_url: Ollama API URL
        
    Returns:
        bool: モデルが利用可能な場合True
    """
    try:
        url = f"{base_url}/api/tags"
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to get model list: {response.status_code}")
            return False
        
        models = response.json().get("models", [])
        available_models = [model["name"] for model in models]
        
        # モデル名のチェック（バージョン指定を含む場合と含まない場合の両方）
        for available in available_models:
            if available == model_name or available.split(":")[0] == model_name.split(":")[0]:
                return True
        
        logger.warning(f"Model {model_name} not found in available models: {available_models}")
        return False
        
    except Exception as e:
        logger.error(f"Error checking model availability: {str(e)}")
        return False

async def generate_book_outline(research_data: str, topic: str, ollama_config: OllamaConfig) -> Dict[str, Any]:
    """
    研究データから本の概要を生成
    
    Args:
        research_data: 収集された研究データ
        topic: 本のトピック
        ollama_config: Ollama設定
        
    Returns:
        Dict: 本の概要（タイトル、章立てなど）
    """
    system_prompt = """あなたは優れた本の企画者です。提供される研究データに基づいて、本の構造を設計してください。
読者にとって価値のある、論理的に整理された本の概要を生成します。"""
    
    # プロンプトの作成
    prompt = f"""以下の研究データに基づいて、「{topic}」に関する本の概要を作成してください。

研究データ:
```
{research_data[:4000]}  # 長すぎる場合は最初の部分のみ使用
```

以下の構造でJSON形式で回答してください:
{{
  "title": "本のタイトル",
  "subtitle": "サブタイトル（あれば）",
  "description": "本の概要説明（200-300文字）",
  "target_audience": "対象読者",
  "chapters": [
    {{
      "title": "第1章タイトル",
      "summary": "章の概要（100-150文字）",
      "sections": ["セクション1", "セクション2", ...]
    }},
    ...
  ],
  "estimated_pages": 推定ページ数
}}

必ず以下の点を守ってください:
1. 章の数は5-10程度にしてください
2. 日本語で回答してください
3. JSONフォーマットで回答してください（改行や適切なインデントを含む）
4. 各章は論理的な流れを持ち、トピックを体系的にカバーするようにしてください
"""
    
    try:
        # 概要の生成
        response = await generate_text_async(prompt, ollama_config, system_prompt)
        
        # JSONの部分を抽出
        try:
            # JSONの開始と終了を見つけて抽出
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                outline = json.loads(json_str)
                return outline
            else:
                # JSON形式でない場合、エラーログを出力
                logger.error(f"Failed to extract JSON from response: {response}")
                # エラー時にはシンプルな構造を返す
                return {
                    "title": f"{topic}の基本ガイド",
                    "subtitle": "",
                    "description": f"{topic}について解説する本です。",
                    "target_audience": "初心者から中級者",
                    "chapters": [
                        {"title": "イントロダクション", "summary": f"{topic}の基本", "sections": ["概要", "歴史"]}
                    ],
                    "estimated_pages": 50
                }
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}\nResponse: {response}")
            # エラー時にはシンプルな構造を返す
            return {
                "title": f"{topic}の基本ガイド",
                "subtitle": "",
                "description": f"{topic}について解説する本です。",
                "target_audience": "初心者から中級者",
                "chapters": [
                    {"title": "イントロダクション", "summary": f"{topic}の基本", "sections": ["概要", "歴史"]}
                ],
                "estimated_pages": 50
            }
    
    except Exception as e:
        logger.error(f"Error generating book outline: {str(e)}")
        # エラー時にはシンプルな構造を返す
        return {
            "title": f"{topic}の基本ガイド",
            "subtitle": "",
            "description": f"{topic}について解説する本です。",
            "target_audience": "初心者から中級者",
            "chapters": [
                {"title": "イントロダクション", "summary": f"{topic}の基本", "sections": ["概要", "歴史"]}
            ],
            "estimated_pages": 50
        }

async def generate_chapter_content(
    chapter_info: Dict[str, Any], 
    topic: str, 
    research_data: str,
    outline: Dict[str, Any],
    ollama_config: OllamaConfig
) -> Dict[str, Any]:
    """
    章の内容を生成
    
    Args:
        chapter_info: 章の情報
        topic: 本のトピック
        research_data: 収集された研究データ
        outline: 本全体の概要
        ollama_config: Ollama設定
        
    Returns:
        Dict: 生成された章のコンテンツ
    """
    chapter_title = chapter_info["title"]
    chapter_summary = chapter_info["summary"]
    sections = chapter_info.get("sections", [])
    
    system_prompt = """あなたは優れた技術書作家です。研究データとアウトラインに基づいて、章の内容を書いてください。
情報が正確で、読みやすく、教育的な内容を生成してください。適切な見出し、段落分け、例示を含めてください。"""
    
    # プロンプトの作成
    prompt = f"""「{topic}」に関する本の「{chapter_title}」の章を書いてください。
この章の概要: {chapter_summary}

本の全体的な構造:
タイトル: {outline["title"]}
概要: {outline["description"]}

以下の研究データを参考にしてください:
```
{research_data[:3000]}  # 長すぎる場合は最初の部分のみ使用
```

以下のセクションを含めてください:
{', '.join(sections)}

以下の形式でマークダウン形式で回答してください:
```markdown
# {chapter_title}

## セクション1
内容...

## セクション2
内容...

...
```

特に以下に注意してください:
1. 冒頭で章の概要を簡潔に述べてください
2. 研究データの事実情報のみに基づき、事実に基づかない情報は含めないでください
3. 専門用語は初めて登場する際に説明してください
4. 読者がトピックを理解しやすいように、例や説明を含めてください
5. 段落ごとにひとつの主要な概念に焦点を当ててください
"""
    
    try:
        # 章の内容を生成
        content = await generate_text_async(prompt, ollama_config, system_prompt)
        
        # マークダウンの本文以外を削除
        markdown_start = content.find('# ')
        if markdown_start >= 0:
            content = content[markdown_start:]
        
        # 結果を返す
        return {
            "title": chapter_title,
            "summary": chapter_summary,
            "content": content,
            "sections": sections
        }
    
    except Exception as e:
        logger.error(f"Error generating chapter content: {str(e)}")
        # エラー時には基本的な内容を返す
        return {
            "title": chapter_title,
            "summary": chapter_summary,
            "content": f"# {chapter_title}\n\n{chapter_summary}\n\n内容の生成中にエラーが発生しました。",
            "sections": sections
        }

# メインの部分（テスト用）
if __name__ == "__main__":
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # テスト関数
    async def test_ollama():
        # 設定
        config = OllamaConfig(model="gemma3:4b", temperature=0.7)
        
        # モデルの可用性をチェック
        available = await check_model_availability(config.model)
        print(f"Model {config.model} available: {available}")
        
        if available:
            # テキスト生成のテスト
            prompt = "Pythonプログラミングの主な特徴を3つ挙げて、それぞれ説明してください。"
            
            # 非同期生成
            response = await generate_text_async(prompt, config)
            print("\nGenerated text (async):")
            print(response)
    
    # テスト実行
    asyncio.run(test_ollama()) 