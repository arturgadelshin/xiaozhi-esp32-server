import os
import wave
from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase
from piper.voice import PiperVoice

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.model_path = config.get("model_path", "models/tts/ru_RU-dmitri-medium.onnx")
        self.config_path = config.get("config_path", "models/tts/ru_RU-dmitri-medium.onnx.json")
        self.output_dir = config.get("output_dir", "tmp/")
        self.delete_audio_file = delete_audio_file

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Модель TTS не найдена: {self.model_path}")
        if not os.path.isfile(self.config_path):
            logger.bind(tag=TAG).warning(f"Файл конфигурации не найден: {self.config_path}")

        os.makedirs(self.output_dir, exist_ok=True)

        try:
            self.voice = PiperVoice.load(
                model_path=self.model_path,
                config_path=self.config_path,
                use_cuda=False
            )
            logger.bind(tag=TAG).info("✅ Piper TTS успешно инициализирован")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Ошибка загрузки модели Piper: {e}")
            raise

    async def text_to_speak(self, text: str, output_file: str) -> bytes:
        try:
            if output_file is None:
                output_file = self.generate_filename(".wav")

            # Создаем WAV-файл и записываем чанки
            with wave.open(output_file, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.voice.config.sample_rate)

                for audio_chunk in self.voice.synthesize(text):
                    audio_int16 = (audio_chunk.audio_float_array * 32767).astype('int16').tobytes()
                    wav_file.writeframes(audio_int16)

            # Читаем аудиофайл
            with open(output_file, "rb") as f:
                audio_data = f.read()

            # Удаляем временный файл, если нужно
            if self.delete_audio_file and os.path.exists(output_file):
                os.remove(output_file)

            # Проверяем размер аудио
            if len(audio_data) == 0:
                raise ValueError("Сгенерированный аудиофайл пуст")

            # Возвращаем bytes, а не list
            # audio_data_list = list(audio_data)  # ❌ Было
            logger.bind(tag=TAG).info(f"✅ TTS сгенерирован: {text[:50]}..., размер: {len(audio_data)} байт")
            return audio_data  # ✅ Стало: возвращаем bytes

        except Exception as e:
            logger.bind(tag=TAG).error(f"Ошибка TTS: {e}")
            raise