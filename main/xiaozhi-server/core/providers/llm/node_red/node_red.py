import json
import requests
from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase
from core.providers.llm.system_prompt import get_system_prompt_for_function
from core.utils.util import check_model_key

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "http://localhost:1880/dialog").rstrip("/")
        model_key_msg = check_model_key("NodeRedLLM", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

    def response(self, session_id, dialogue, **kwargs):
        try:
            # Отправляем весь диалог целиком
            payload = {
                "session_id": session_id,
                "dialogue": dialogue  # Всё диалоговое окружение
            }
            headers = {"Content-Type": "application/json"}

            with requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            ) as r:
                if r.status_code != 200:
                    logger.bind(tag=TAG).error(f"Node-RED request failed: {r.status_code}")
                    yield "【服务响应异常】"
                    return

                # Ожидаем ответ в формате JSON с полем 'answer'
                try:
                    response_data = r.json()
                    answer = response_data.get("answer", "").strip()
                    if answer:
                        yield answer
                    else:
                        logger.bind(tag=TAG).error("Node-RED returned empty or invalid answer")
                        yield "Извините, я не могу ответить на ваш запрос."
                except json.JSONDecodeError:
                    # Если ответ не JSON, отправляем как есть
                    yield r.text.strip()

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in response generation: {e}")
            yield "【服务响应异常】"

    def response_with_functions(self, session_id, dialogue, functions=None):
        # Если есть функции, добавляем системный промпт
        if len(dialogue) == 2 and functions is not None and len(functions) > 0:
            last_msg = dialogue[-1]["content"]
            function_str = json.dumps(functions, ensure_ascii=False)
            modify_msg = get_system_prompt_for_function(function_str) + last_msg
            dialogue[-1]["content"] = modify_msg

        # Обрабатываем результат вызова функции
        if len(dialogue) > 1 and dialogue[-1]["role"] == "tool":
            assistant_msg = "\ntool call result: " + dialogue[-1]["content"] + "\n\n"
            while len(dialogue) > 1:
                if dialogue[-1]["role"] == "user":
                    dialogue[-1]["content"] = assistant_msg + dialogue[-1]["content"]
                    break
                dialogue.pop()

        # Передаём обновлённый диалог
        for token in self.response(session_id, dialogue):
            yield token, None