import sherpa_onnx

print("✅ sherpa-onnx импортирован")

# Проверим, есть ли метод
if hasattr(sherpa_onnx.OfflineRecognizer, "from_zipformer"):
    print("✅ Метод from_zipformer ДОСТУПЕН")
else:
    print("❌ Метод from_zipformer отсутствует — нужен апдейт")