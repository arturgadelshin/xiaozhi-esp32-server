import time
import wave
import os
import sys
import io
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

import numpy as np
import sherpa_onnx

TAG = __name__
logger = setup_logging()


# Класс для подавления вывода (например, от ONNX)
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        output = self._output.getvalue()
        self._output.close()
        if output.strip():
            logger.bind(tag=TAG).info(output.strip())


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        self.interface_type = InterfaceType.LOCAL
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")
        self.model_type = config.get("model_type", "transducer")
        self.delete_audio_file = delete_audio_file

        # Создаём папку для временных файлов
        os.makedirs(self.output_dir, exist_ok=True)

        # Пути к модели
        self.encoder_path = os.path.join(self.model_dir, "encoder.int8.onnx")
        self.decoder_path = os.path.join(self.model_dir, "decoder.int8.onnx")
        self.joiner_path = os.path.join(self.model_dir, "joiner.int8.onnx")
        self.tokens_path = os.path.join(self.model_dir, "tokens.txt")

        # Проверка наличия всех необходимых файлов
        for path, name in [
            (self.encoder_path, "Encoder"),
            (self.decoder_path, "Decoder"),
            (self.joiner_path, "Joiner"),
            (self.tokens_path, "Tokens"),
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name}-файл не найден: {path}")

        # Загрузка модели
        with CaptureOutput():
            try:
                self.model = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=self.encoder_path,
                    decoder=self.decoder_path,
                    joiner=self.joiner_path,
                    tokens=self.tokens_path,
                    num_threads=4,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    debug=False,
                )
                logger.bind(tag=TAG).info("✅ Модель ASR успешно загружена")
            except Exception as e:
                logger.bind(tag=TAG).error(f"Ошибка загрузки модели: {e}")
                raise

        logger.bind(tag=TAG).info("✅ Модель ASR успешно загружена и готова к работе.")

    def read_wave(self, wave_filename: str) -> tuple[np.ndarray, int]:
        with wave.open(wave_filename) as f:
            assert f.getnchannels() == 1, f.getnchannels()
            assert f.getsampwidth() == 2, f.getsampwidth()
            num_samples = f.getnframes()
            samples = f.readframes(num_samples)
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0
            return samples_float32, f.getframerate()

    async def speech_to_text(
            self, opus_data: list[bytes], session_id: str, audio_format="opus"
    ) -> tuple[str | None, str | None]:
        file_path = None
        try:
            # Декодируем Opus → PCM
            from core.utils.util import decode_opus
            pcm_data = decode_opus(opus_data)

            if not pcm_data:
                logger.bind(tag=TAG).error("Пустые аудиоданные после декодирования")
                return "", None

            # Сохраняем временный файл
            file_path = self.save_audio_to_file(pcm_data, session_id)

            # Распознавание
            start_time = time.time()
            stream = self.model.create_stream()
            samples, sample_rate = self.read_wave(file_path)
            stream.accept_waveform(sample_rate, samples)
            self.model.decode_stream(stream)
            text = stream.result.text.strip()
            logger.bind(tag=TAG).debug(f"Распознано: '{text}' за {time.time() - start_time:.3f} с")
            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"Ошибка ASR: {e}")
            return "", file_path
        finally:
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.bind(tag=TAG).error(f"Не удалось удалить файл: {e}")

    def transcribe_pcm(self, pcm_data: bytes, sample_rate: int = 16000) -> str:
        """
        Распознавание PCM-аудио (16-бит, моно) без сохранения в файл.
        """
        if not pcm_data or len(pcm_data) == 0:
            return ""

        # Убедимся, что длина кратна 2 (16 бит = 2 байта)
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]

        try:
            # Конвертируем байты в float32 [-1.0, 1.0]
            samples_int16 = np.frombuffer(pcm_data, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0

            # Проверка частоты
            if sample_rate != 16000:
                logger.bind(tag=TAG).warning(
                    f"Частота {sample_rate} Гц не равна 16 кГц. Распознавание может быть неточным."
                )

            # Создаём поток и распознаём
            stream = self.model.create_stream()
            stream.accept_waveform(16000, samples_float32)
            self.model.decode_stream(stream)
            text = stream.result.text.strip()
            return text

        except Exception as e:
            logger.bind(tag=TAG).error(f"Ошибка при распознавании PCM: {e}")
            return ""

    def save_audio_to_file(self, pcm_data: bytes, session_id: str) -> str:
        """Сохраняет PCM в .wav (для отладки или если нужно)"""
        file_path = os.path.join(self.output_dir, f"{session_id}.wav")
        with wave.open(file_path, "wb") as f:
            f.setnchannels(1)  # моно
            f.setsampwidth(2)  # 16 бит = 2 байта
            f.setframerate(16000)
            f.writeframes(pcm_data)
        return file_path

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        """Основной метод: Opus → PCM → Распознавание"""
        file_path = None
        try:
            start_time = time.time()

            # Декодируем Opus в PCM
            if audio_format == "pcm":
                pcm_data = b"".join(opus_data)
            else:
                from core.utils.util import decode_opus  # Убедись, что у тебя есть эта функция
                pcm_data = decode_opus(opus_data)

            if not pcm_data:
                logger.bind(tag=TAG).error("Пустые аудиоданные после декодирования")
                return "", None

            # Сохраняем только если нужно (например, для отладки)
            if self.delete_audio_file:
                file_path = self.save_audio_to_file(pcm_data, session_id)

            # Распознаём напрямую из PCM
            text = self.transcribe_pcm(pcm_data, sample_rate=16000)

            logger.bind(tag=TAG).debug(
                f"Распознавание завершено за {time.time() - start_time:.3f} с | Текст: '{text}'"
            )
            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"Ошибка распознавания: {e}", exc_info=True)
            return "", file_path

        finally:
            # Удаляем временный файл
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.bind(tag=TAG).debug(f"Временный файл удалён: {file_path}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"Не удалось удалить файл {file_path}: {e}")