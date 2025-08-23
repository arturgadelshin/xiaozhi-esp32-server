import json
import time
from core.handle.abortHandle import handleAbortMessage
from core.handle.helloHandle import handleHelloMessage
from core.providers.tools.device_mcp import handle_mcp_message
from core.utils.util import remove_punctuation_and_length, filter_sensitive_info
from core.handle.receiveAudioHandle import startToChat, handleAudioMessage
from core.handle.sendAudioHandle import send_stt_message, send_tts_message
from core.providers.tools.device_iot import handleIotDescriptors, handleIotStatus
from core.handle.reportHandle import enqueue_asr_report
import asyncio

TAG = __name__


async def handleTextMessage(conn, message):
    """Обрабатывает текстовые сообщения от клиента (ESP32)"""
    try:
        msg_json = json.loads(message)

        # Если сообщение — просто число (редкий случай)
        if isinstance(msg_json, int):
            conn.logger.bind(tag=TAG).info(f"Получено текстовое сообщение: {message}")
            await conn.websocket.send(message)
            return

        # Приветствие — инициализация клиента
        if msg_json["type"] == "hello":
            conn.logger.bind(tag=TAG).info(f"Получено hello-сообщение: {message}")
            await handleHelloMessage(conn, msg_json)

        # Прерывание текущего ответа
        elif msg_json["type"] == "abort":
            conn.logger.bind(tag=TAG).info(f"Получено abort-сообщение: {message}")
            await handleAbortMessage(conn)

        # Событие "слушать" — начало/остановка записи
        elif msg_json["type"] == "listen":
            conn.logger.bind(tag=TAG).info(f"Получено listen-сообщение: {message}")

            # Режим захвата звука (ручной/авто)
            if "mode" in msg_json:
                conn.client_listen_mode = msg_json["mode"]
                conn.logger.bind(tag=TAG).debug(f"Режим микрофона: {conn.client_listen_mode}")

            # Начало записи — очищаем буфер
            if msg_json["state"] == "start":
                conn.client_have_voice = True
                conn.client_voice_stop = False
                conn.asr_audio.clear()  # Очищаем предыдущие данные
                conn.logger.bind(tag=TAG).debug("🎙️ Начало записи. Буфер аудио очищен.")

            # Окончание записи — запускаем распознавание
            elif msg_json["state"] == "stop":
                conn.client_have_voice = True
                conn.client_voice_stop = True
                conn.logger.bind(tag=TAG).debug(f"⏹️ Запись остановлена. Размер буфера: {len(conn.asr_audio)} байт")

                # Если есть аудио — обрабатываем
                if len(conn.asr_audio) > 0:
                    await handleAudioMessage(conn, b"")  # Передаём пустые данные, но буфер уже заполнен
                else:
                    conn.logger.bind(tag=TAG).warning("❌ Аудио не получено — буфер пуст.")

            # Режим детекции (например, wake-word)
            elif msg_json["state"] == "detect":
                conn.client_have_voice = False
                conn.asr_audio.clear()

                if "text" in msg_json:
                    original_text = msg_json["text"]
                    filtered_len, filtered_text = remove_punctuation_and_length(original_text)
                    is_wakeup_words = filtered_text in conn.config.get("wakeup_words", [])
                    enable_greeting = conn.config.get("enable_greeting", True)

                    if is_wakeup_words and not enable_greeting:
                        await send_stt_message(conn, original_text)
                        await send_tts_message(conn, "stop", None)
                        conn.client_is_speaking = False
                    elif is_wakeup_words:
                        conn.just_woken_up = True
                        enqueue_asr_report(conn, "Привет!", [])
                        await startToChat(conn, "Привет!")
                    else:
                        enqueue_asr_report(conn, original_text, [])
                        await startToChat(conn, original_text)

        # Обработка IoT-устройств
        elif msg_json["type"] == "iot":
            conn.logger.bind(tag=TAG).info(f"Получено iot-сообщение: {message}")
            if "descriptors" in msg_json:
                asyncio.create_task(handleIotDescriptors(conn, msg_json["descriptors"]))
            if "states" in msg_json:
                asyncio.create_task(handleIotStatus(conn, msg_json["states"]))

        # Обработка MCP (инструменты)
        elif msg_json["type"] == "mcp":
            conn.logger.bind(tag=TAG).info(f"Получено mcp-сообщение: {message[:100]}")
            if "payload" in msg_json:
                asyncio.create_task(handle_mcp_message(conn, conn.mcp_client, msg_json["payload"]))

        # Управление сервером (API)
        elif msg_json["type"] == "server":
            conn.logger.bind(tag=TAG).info(f"Получено server-сообщение: {filter_sensitive_info(msg_json)}")

            if not conn.read_config_from_api:
                return

            post_secret = msg_json.get("content", {}).get("secret", "")
            secret = conn.config["manager-api"].get("secret", "")

            if post_secret != secret:
                await conn.websocket.send(json.dumps({
                    "type": "server",
                    "status": "error",
                    "message": "Ошибка проверки ключа"
                }))
                return

            if msg_json["action"] == "update_config":
                try:
                    if not conn.server or not await conn.server.update_config():
                        await conn.websocket.send(json.dumps({
                            "type": "server",
                            "status": "error",
                            "message": "Ошибка обновления конфигурации"
                        }))
                        return
                    await conn.websocket.send(json.dumps({
                        "type": "server",
                        "status": "success",
                        "message": "Конфигурация обновлена"
                    }))
                except Exception as e:
                    conn.logger.bind(tag=TAG).error(f"Ошибка обновления конфигурации: {e}")
                    await conn.websocket.send(json.dumps({
                        "type": "server",
                        "status": "error",
                        "message": f"Ошибка: {str(e)}"
                    }))

            elif msg_json["action"] == "restart":
                await conn.handle_restart(msg_json)

        else:
            conn.logger.bind(tag=TAG).error(f"Неизвестный тип сообщения: {message}")

    except json.JSONDecodeError:
        await conn.websocket.send(message)