# core/providers/asr/vosk.py

import os
import numpy as np
from typing import Optional, Tuple, List
from vosk import Model as VoskModel, KaldiRecognizer
import json

from config.logger import setup_logging
from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool = True):
        super().__init__()
        self.interface_type = InterfaceType.LOCAL
        self.delete_audio_file = delete_audio_file
        model_dir = config.get("model_dir")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Модель Vosk не найдена: {model_dir}")

        try:
            self.model = VoskModel(model_dir)
            self.recognizer = None
            logger.bind(tag=TAG).info("✅ Vosk: модель загружена")
        except Exception as e:
            logger.bind(tag=TAG).error(f"❌ Ошибка загрузки Vosk: {e}")
            self.model = None

    def transcribe(self, pcm_data: bytes) -> str:
        if not self.model or not pcm_data:
            return ""

        try:
            if len(pcm_data) % 2 != 0:
                pcm_data = pcm_data[:-1]

            rec = KaldiRecognizer(self.model, 16000)
            if rec.AcceptWaveform(pcm_data):
                result = rec.Result()
            else:
                result = rec.PartialResult()

            text = json.loads(result).get("text", "").strip()
            return text
        except Exception as e:
            logger.bind(tag=TAG).error(f"❌ Ошибка распознавания Vosk: {e}")
            return ""

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        total_bytes = sum(len(chunk) for chunk in opus_data)
        logger.bind(tag=TAG).info("ASR processing audio: %d bytes", total_bytes)

        if total_bytes == 0:
            logger.bind(tag=TAG).error("Пустые аудиоданные")
            return "", None

        try:
            from core.utils.util import decode_opus
            pcm_data = decode_opus(opus_data) if audio_format != "pcm" else b"".join(opus_data)
        except Exception as e:
            logger.bind(tag=TAG).error(f"Ошибка декодирования: {e}")
            return "", None

        if not pcm_data:
            logger.bind(tag=TAG).error("PCM пустой после декодирования")
            return "", None

        text = self.transcribe(pcm_data)

        if text:
            logger.bind(tag=TAG).debug(f"✅ Vosk: распознано — '{text}'")
        else:
            logger.bind(tag=TAG).warning("❌ Vosk: ничего не распознано")

        return text, None