import chainlit as cl
import ollama
from typing import Dict, List, Any
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# システムメッセージの設定
SYSTEM_MESSAGE = "You are a helpful assistant."
# MODEL_NAME = "gemma3:4b"
MODEL_NAME = "gemma3:27b"

@cl.on_chat_start
async def start_chat():
    """チャットセッション開始時の処理"""
    try:
        # ユーザーセッションに初期対話履歴を設定
        cl.user_session.set(
            "interaction",
            [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                }
            ],
        )
        
        # 空のメッセージオブジェクトを作成
        msg = cl.Message(content="")
        
        # 開始メッセージを定義
        start_message = """Hello, I'm your 100% local ChatGPT powered by Google DeepMind's Gemma 3. How can I help you today?"""
        
        # 開始メッセージをトークンごとにストリーミング
        for token in start_message:
            await msg.stream_token(token)
        
        # メッセージを送信
        await msg.send()
        
    except Exception as e:
        logger.error(f"Error in start_chat: {e}")
        await cl.Message(content=f"Error starting chat: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """ユーザーからのメッセージを処理する関数"""
    try:
        # ツール関数を呼び出して応答を取得
        tool_res = await tool(message.content)
        
        # 空のメッセージオブジェクトを作成
        msg = cl.Message(content="")
        
        # 応答をトークンごとにストリーミング
        for token in tool_res.message.content:
            await msg.stream_token(token)
        
        # メッセージを送信
        await msg.send()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        await cl.Message(content=f"Error processing your message: {str(e)}").send()

@cl.step(type="tool")
async def tool(input_message: str):
    """AIモデルを使用して応答を生成するツール関数"""
    try:
        # ユーザーセッションから対話履歴を取得
        interaction = cl.user_session.get("interaction")
        
        # ユーザーのメッセージを対話履歴に追加
        interaction.append({
            "role": "user",
            "content": input_message
        })
        
        # Ollamaを使用して応答を生成
        response = ollama.chat(
            model=MODEL_NAME,
            messages=interaction
        )
        
        # 生成された応答を対話履歴に追加
        interaction.append({
            "role": "assistant",
            "content": response.message.content
        })
        
        # 応答を返す
        return response
        
    except Exception as e:
        logger.error(f"Error in tool function: {e}")
        raise