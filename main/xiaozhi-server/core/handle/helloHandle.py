import time
import json
import random
import asyncio
from core.utils.dialogue import Message
from core.utils.util import audio_to_data
from core.handle.sendAudioHandle import sendAudioMessage, send_stt_message
from core.utils.util import remove_punctuation_and_length, opus_datas_to_wav_bytes
from core.providers.tts.dto.dto import ContentType, SentenceType
from core.providers.tools.device_mcp import (
    MCPClient,
    send_mcp_initialize_message,
    send_mcp_tools_list_request,
)
from core.utils.wakeup_word import WakeupWordsConfig

TAG = __name__

# Конфигурация активации по ключевым словам
WAKEUP_CONFIG = {
    "refresh_time": 5,  # Время (сек), после которого можно обновить приветствие
    "words": ["привет", "эй", "ассистент", "слушай", "будь добр"],  # Поддерживаемые wake-слова
}

# Глобальный менеджер настроек wake-слов
wakeup_words_config = WakeupWordsConfig()

# Блокировка для предотвращения одновременного вызова wakeupWordsResponse
_wakeup_response_lock = asyncio.Lock()


async def handleHelloMessage(conn, msg_json):
    """Обрабатывает сообщение 'hello' от клиента"""
    audio_params = msg_json.get("audio_params")
    if audio_params:
        format = audio_params.get("format")
        conn.logger.bind(tag=TAG).info(f"Формат аудио клиента: {format}")
        conn.audio_format = format
        conn.welcome_msg["audio_params"] = audio_params

    features = msg_json.get("features")
    if features:
        conn.logger.bind(tag=TAG).info(f"Возможности клиента: {features}")
        conn.features = features
        if features.get("mcp"):
            conn.logger.bind(tag=TAG).info("Клиент поддерживает MCP")
            conn.mcp_client = MCPClient()
            # Отправить инициализацию
            asyncio.create_task(send_mcp_initialize_message(conn))
            # Запросить список инструментов
            asyncio.create_task(send_mcp_tools_list_request(conn))

    await conn.websocket.send(json.dumps(conn.welcome_msg))


async def checkWakeupWords(conn, text):
    """Проверяет, произнесено ли ключевое слово, и отвечает"""
    enable_wakeup_words_response_cache = conn.config.get("enable_wakeup_words_response_cache")

    if not enable_wakeup_words_response_cache or not conn.tts:
        return False

    _, filtered_text = remove_punctuation_and_length(text)
    if filtered_text not in conn.config.get("wakeup_words", []):
        return False

    conn.just_woken_up = True
    await send_stt_message(conn, text)

    # Получаем текущий голос
    voice = getattr(conn.tts, "voice", "default")
    if not voice:
        voice = "default"

    # Получаем настройку ответа на пробуждение
    response = wakeup_words_config.get_wakeup_response(voice)
    if not response or not response.get("file_path"):
        response = {
            "voice": "default",
            "file_path": "config/assets/wakeup_words.wav",
            "time": 0,
            "text": "Привет! Я — голосовой помощник. Готов помочь с умным домом или просто поболтать!",
        }

    # Проигрываем аудио-ответ
    conn.client_abort = False
    opus_packets, _ = audio_to_data(response.get("file_path"))

    conn.logger.bind(tag=TAG).info(f"Проигрывается приветствие: {response.get('text')}")
    await sendAudioMessage(conn, SentenceType.FIRST, opus_packets, response.get("text"))
    await sendAudioMessage(conn, SentenceType.LAST, [], None)

    # Добавляем в историю диалога
    conn.dialogue.put(Message(role="assistant", content=response.get("text")))

    # Проверяем, нужно ли обновить приветствие
    if time.time() - response.get("time", 0) > WAKEUP_CONFIG["refresh_time"]:
        if not _wakeup_response_lock.locked():
            asyncio.create_task(wakeupWordsResponse(conn))
    return True


async def wakeupWordsResponse(conn):
    """Генерирует новое уникальное приветствие через LLM + TTS"""
    if not conn.tts or not conn.llm or not conn.llm.response_no_stream:
        return

    try:
        # Пытаемся получить блокировку
        if not await _wakeup_response_lock.acquire():
            return

        # Генерируем случайное ключевое слово
        wakeup_word = random.choice(WAKEUP_CONFIG["words"])
        question = (
            f"Пользователь сказал: `{wakeup_word}`.\n"
            "Ответь в 20–30 слов на русском языке. Будь дружелюбным, живым, как настоящий человек.\n"
            "Не объясняй свой ответ, не используй эмодзи. Только текст ответа."
        )

        result = conn.llm.response_no_stream(conn.config["prompt"], question)
        if not result or len(result.strip()) == 0:
            return

        # Генерируем аудио через TTS
        tts_result = await asyncio.to_thread(conn.tts.to_tts, result)
        if not tts_result:
            return

        # Получаем текущий голос
        voice = getattr(conn.tts, "voice", "default")

        wav_bytes = opus_datas_to_wav_bytes(tts_result, sample_rate=16000)
        file_path = wakeup_words_config.generate_file_path(voice)
        with open(file_path, "wb") as f:
            f.write(wav_bytes)

        # Обновляем настройку приветствия
        wakeup_words_config.update_wakeup_response(voice, file_path, result)

    finally:
        # Гарантированно освобождаем блокировку
        if _wakeup_response_lock.locked():
            _wakeup_response_lock.release()