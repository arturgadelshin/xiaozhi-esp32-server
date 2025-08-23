# core/providers/llm/ollama/ollama.py

from config.logger import setup_logging
from openai import OpenAI
import json
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.base_url = config.get("base_url", "http://localhost:11434")

        # Добавляем /v1, если нет
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama",  # Ollama не требует ключ, но OpenAI client требует
        )

        # Проверяем, модель ли это qwen3
        self.is_qwen3 = self.model_name and self.model_name.lower().startswith("qwen3")

    def response(self, session_id, dialogue, **kwargs):
        """
        Генерация текстового ответа (без инструментов)
        """
        try:
            # 🔹 Логируем входные данные
            logger.bind(tag=TAG).debug(f"📝 Начало генерации ответа для сессии: {session_id}")
            logger.bind(tag=TAG).debug(f"💬 Диалог (вход): {dialogue}")

            # 🔥 ЛОГИРУЕМ ПОЛНЫЙ ПРОМПТ, КОТОРЫЙ ПОЙДЁТ В OLLAMA
            full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in dialogue])
            logger.bind(tag=TAG).info(f"📤 ПОЛНЫЙ ПРОМПТ ДО OLLAMA:\n{full_prompt}")

            # Если это qwen3 — добавляем /no_think
            if self.is_qwen3:
                dialogue_copy = dialogue.copy()
                for i in range(len(dialogue_copy) - 1, -1, -1):
                    if dialogue_copy[i]["role"] == "user":
                        dialogue_copy[i]["content"] = "/no_think " + dialogue_copy[i]["content"]
                        logger.bind(tag=TAG).debug(f"🔧 Добавлен /no_think для qwen3")
                        break
                dialogue = dialogue_copy

            # 🔹 Генерируем потоковый ответ
            responses = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=True
            )

            is_active = True  # Флаг: внутри <think> или нет
            buffer = ""       # Буфер для сбора текста

            for chunk in responses:
                try:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    content = delta.content if hasattr(delta, "content") else ""
                    finish_reason = chunk.choices[0].finish_reason if hasattr(chunk.choices[0], "finish_reason") else None

                    # 🔹 Логируем сырой чанк
                    logger.bind(tag=TAG).trace(f"📦 Получен чанк: content='{content}', finish_reason='{finish_reason}'")

                    if content:
                        buffer += content

                        # Обработка тегов <think>
                        while "<think>" in buffer and "</think>" in buffer:
                            pre = buffer.split("<think>", 1)[0]
                            post = buffer.split("</think>", 1)[1]
                            buffer = pre + post
                            logger.bind(tag=TAG).debug("🗑️ Удалён блок <think>...<think>")

                        if "<think>" in buffer:
                            is_active = False
                            buffer = buffer.split("<think>", 1)[0]
                            logger.bind(tag=TAG).debug("⏸️ Вход в режим <think> (текст не озвучивается)")

                        if "</think>" in buffer:
                            is_active = True
                            buffer = buffer.split("</think>", 1)[1]
                            logger.bind(tag=TAG).debug("▶️ Выход из режима <think> (озвучка возобновлена)")

                        # Отправляем только активный текст
                        if is_active and buffer.strip():
                            logger.bind(tag=TAG).info(f"🔊 Выходной чанк: '{buffer}'")
                            yield buffer
                            buffer = ""

                    # Конец генерации
                    if finish_reason:
                        if buffer.strip():
                            logger.bind(tag=TAG).info(f"🔚 Последний чанк: '{buffer}'")
                            yield buffer
                        logger.bind(tag=TAG).debug(f"🏁 Генерация завершена (причина: {finish_reason})")
                        break

                except Exception as e:
                    logger.bind(tag=TAG).error(f"❌ Ошибка обработки чанка: {e}")
                    continue

        except Exception as e:
            error_msg = "【Ollama: Ошибка соединения】"
            logger.bind(tag=TAG).error(f"❌ Ошибка генерации ответа: {e}")
            yield error_msg

    def response_with_functions(self, session_id, dialogue, functions=None):
        """
        Генерация с поддержкой вызова инструментов (tools)
        """
        try:
            logger.bind(tag=TAG).debug(f"🔧 Начало генерации с инструментами для сессии: {session_id}")
            logger.bind(tag=TAG).debug(f"💬 Диалог: {dialogue}")
            if functions:
                logger.bind(tag=TAG).debug(f"🛠️ Доступные инструменты: {[f['function']['name'] for f in functions]}")

            # 🔥 ЛОГИРУЕМ ПОЛНЫЙ ПРОМПТ С ИНСТРУМЕНТАМИ
            full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in dialogue])
            logger.bind(tag=TAG).info(f"📤 ПОЛНЫЙ ПРОМПТ С ИНСТРУМЕНТАМИ:\n{full_prompt}")
            if functions:
                logger.bind(tag=TAG).info(f"🛠️ ЗАПРОШЕННЫЕ ИНСТРУМЕНТЫ: {[f['function']['name'] for f in functions]}")

            # Для qwen3 добавляем /no_think
            if self.is_qwen3:
                dialogue_copy = dialogue.copy()
                for i in range(len(dialogue_copy) - 1, -1, -1):
                    if dialogue_copy[i]["role"] == "user":
                        dialogue_copy[i]["content"] = "/no_think " + dialogue_copy[i]["content"]
                        logger.bind(tag=TAG).debug("🔧 Добавлен /no_think для qwen3")
                        break
                dialogue = dialogue_copy

            # Отправляем запрос с tools
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=True,
                tools=functions,
            )

            is_active = True
            buffer = ""

            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    content = delta.content if hasattr(delta, "content") else None
                    tool_calls = delta.tool_calls if hasattr(delta, "tool_calls") else None
                    finish_reason = chunk.choices[0].finish_reason if hasattr(chunk.choices[0], "finish_reason") else None

                    logger.bind(tag=TAG).trace(f"📦 Чанк: content='{content}', tool_calls={tool_calls}, finish_reason='{finish_reason}'")

                    # 🔧 Сначала проверяем tool_calls
                    if tool_calls:
                        logger.bind(tag=TAG).info(f"🛠️ Вызов инструмента: {tool_calls}")
                        yield None, tool_calls
                        continue

                    # 📝 Обработка текста
                    if content:
                        buffer += content

                        while "<think>" in buffer and "</think>" in buffer:
                            pre = buffer.split("<think>", 1)[0]
                            post = buffer.split("</think>", 1)[1]
                            buffer = pre + post

                        if "<think>" in buffer:
                            is_active = False
                            buffer = buffer.split("<think>", 1)[0]

                        if "</think>" in buffer:
                            is_active = True
                            buffer = buffer.split("</think>", 1)[1]

                        if is_active and buffer.strip():
                            logger.bind(tag=TAG).info(f"🔊 Текстовый чанк: '{buffer}'")
                            yield buffer, None
                            buffer = ""

                    if finish_reason and buffer.strip():
                        logger.bind(tag=TAG).info(f"🔚 Финальный текст: '{buffer}'")
                        yield buffer, None

                except Exception as e:
                    logger.bind(tag=TAG).error(f"❌ Ошибка в потоке с функциями: {e}")
                    continue

        except Exception as e:
            error_msg = f"【Ollama: Ошибка вызова инструментов: {str(e)}】"
            logger.bind(tag=TAG).error(error_msg)
            yield error_msg, None