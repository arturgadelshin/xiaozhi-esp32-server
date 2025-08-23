from piper.voice import PiperVoice, AudioChunk
import wave

# Пути к модели
model_path = "models/tts/ru_RU-dmitri-medium.onnx"
config_path = "models/tts/ru_RU-dmitri-medium.onnx.json"

# Текст для озвучивания
text = "Привет, это тест синтеза речи на русском языке."

# Загружаем модель
voice = PiperVoice.load(
    model_path=model_path,
    config_path=config_path,
    use_cuda=False  # Установите True, если есть GPU
)

# Создаем WAV-файл
with wave.open("output.wav", "wb") as wav_file:
    # Настраиваем параметры файла
    wav_file.setnchannels(1)  # Моно
    wav_file.setsampwidth(2)  # 16 бит (2 байта)
    wav_file.setframerate(voice.config.sample_rate)  # Частота дискретизации из модели

    # Синтезируем речь и записываем чанки
    for audio_chunk in voice.synthesize(text):
        # Преобразуем float32 в int16
        audio_int16 = (audio_chunk.audio_float_array * 32767).astype('int16').tobytes()
        # Записываем в файл
        wav_file.writeframes(audio_int16)

print("✅ Аудиофайл 'output.wav' успешно создан!")